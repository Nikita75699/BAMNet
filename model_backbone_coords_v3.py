import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pytorch_lightning as pl
import os
import matplotlib.pyplot as plt
import numpy as np 

# -------------------------
# Small utils
# -------------------------

def dice_coef(pred, target, eps: float = 1e-6):
    pred = pred.float()
    target = target.float()
    pred = torch.sigmoid(pred)
    dims = (1, 2, 3)
    inter = (pred * target).sum(dims)
    denom = (pred.sum(dims) + target.sum(dims)).clamp_min(eps)
    return (2 * inter / denom).mean()

def dice_loss(logits, target, eps: float = 1e-6):
    return 1.0 - dice_coef(logits, target, eps)

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.float()
        targets = targets.float()
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal = self.alpha * (1.0 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal.mean()
        if self.reduction == 'sum':
            return focal.sum()
        return focal

def safe_efficientnetv2(encoder_name: str):
    name = encoder_name.lower()
    if 'efficientnet_v2_s' in name:
        try:
            weights = models.EfficientNet_V2_S_Weights.DEFAULT
            backbone = models.efficientnet_v2_s(weights=weights)
        except Exception:
            backbone = models.efficientnet_v2_s(pretrained=True)
    elif 'efficientnet_v2_m' in name:
        try:
            weights = models.EfficientNet_V2_M_Weights.DEFAULT
            backbone = models.efficientnet_v2_m(weights=weights)
        except Exception:
            backbone = models.efficientnet_v2_m(pretrained=True)
    else:
        try:
            weights = models.EfficientNet_V2_L_Weights.DEFAULT
            backbone = models.efficientnet_v2_l(weights=weights)
        except Exception:
            backbone = models.efficientnet_v2_l(pretrained=True)
    return backbone.features

# -------------------------
# Coordinate Attention (CA-Net)
# -------------------------

class CoordinateAttention(nn.Module):
    """Coordinate Attention from CA-Net (fixed pooling/concat)"""
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        mid_channels = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Avg-pool вдоль ширины и высоты
        x_h = x.mean(dim=3, keepdim=True)                 # (n, c, h, 1)
        x_w = x.mean(dim=2, keepdim=True).permute(0,1,3,2)  # (n, c, w, 1) — ТРАНСПОНИРУЕМ ДО concat

        # Конкатенация по "высоте": (n, c, h+w, 1)
        x_cat = torch.cat([x_h, x_w], dim=2)
        x_cat = self.conv1(x_cat)
        x_cat = self.bn1(x_cat)
        x_cat = self.relu(x_cat)

        # Разделяем обратно и возвращаем форму ветки w
        x_h, x_w = torch.split(x_cat, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)                     # (n, c, 1, w)

        attention_h = self.sigmoid(self.conv_h(x_h))
        attention_w = self.sigmoid(self.conv_w(x_w))

        return identity * attention_h * attention_w


# -------------------------
# Improved point loss (focal heatmaps + coord + consistency)
# -------------------------

class ImprovedPointLoss(nn.Module):
    def __init__(self, alpha=0.25, beta=0.1, gamma=2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        
    def forward(self, heatmap_pred, heatmap_target, coords_pred, coords_target, visibility):
        # Каст в FP32 для защиты вычислений
        heatmap_pred = heatmap_pred.float()
        heatmap_target = heatmap_target.float()
        coords_pred = coords_pred.float()
        coords_target = coords_target.float()
        visibility = visibility.float()

        # 1. Focal Heatmap loss
        bce_loss = F.binary_cross_entropy_with_logits(heatmap_pred, heatmap_target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_weight = self.alpha * (1.0 - pt) ** self.gamma
        loss_heatmap = (focal_weight * bce_loss).mean()
        
        # 2. Coordinate loss
        scale_factor = max(float(heatmap_pred.shape[-1]), 1.0)
        norm_pred = coords_pred / scale_factor
        norm_tgt = coords_target / scale_factor
        
        # ИСПРАВЛЕНИЕ: 1e-4 вместо 1e-6. В FP16 1e-6 == 0.0, что давало градиент 1/0 = NaN!
        dist_norm = torch.sqrt(((norm_pred - norm_tgt) ** 2).sum(dim=-1) + 1e-4)
        dist_px = dist_norm * scale_factor
        
        loss_coord = (dist_px * visibility).sum() / visibility.sum().clamp_min(1.0)
        
        # 3. Consistency loss
        heatmap_coords = self._heatmap_to_coords(heatmap_pred)
        consistency_loss = torch.tensor(0.0, device=heatmap_pred.device)
        
        total_loss = loss_heatmap + self.beta * loss_coord + 0.1 * consistency_loss
        return total_loss, loss_heatmap, loss_coord
    
    def _heatmap_to_coords(self, heatmap):
        B, P, H, W = heatmap.shape
        heatmap = heatmap.float()
        heatmap_sigmoid = torch.sigmoid(heatmap)
        
        y_coord, x_coord = torch.meshgrid(
            torch.arange(H, device=heatmap.device, dtype=torch.float32),
            torch.arange(W, device=heatmap.device, dtype=torch.float32),
            indexing='ij'
        )
        
        x_coord = x_coord.view(1, 1, H, W).expand(B, P, -1, -1)
        y_coord = y_coord.view(1, 1, H, W).expand(B, P, -1, -1)
        
        denom = heatmap_sigmoid.sum(dim=(-1, -2), keepdim=True) + 1.0
        weights = heatmap_sigmoid / denom
        
        pred_x = (weights * x_coord).sum(dim=(-1, -2))
        pred_y = (weights * y_coord).sum(dim=(-1, -2))
        
        return torch.stack([pred_x, pred_y], dim=-1)

# -------------------------
# Network blocks
# -------------------------

class PositionAttention(nn.Module):
    """
    Глобальное внимание, но:
    - работает на (H * W) <= max_tokens без даунсемплинга
    - если карта слишком большая, сначала усредняем по пространству,
      считаем внимание на уменьшенной сетке и потом апсемплим обратно.
    Так сильно режем риск OOM.
    """
    def __init__(self, attn_qk_dim: int = 64, attn_v_dim: int = 128, max_tokens: int = 4096):
        super().__init__()
        self.query_conv = nn.LazyConv2d(attn_qk_dim, 1)
        self.key_conv   = nn.LazyConv2d(attn_qk_dim, 1)
        self.value_conv = nn.LazyConv2d(attn_v_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        # максимальное число пространственных позиций, на которых считаем внимание напрямую
        self.max_tokens = max_tokens

    def _attend(self, x, skip):
        """
        Вспомогательный блок: считает attention на данных x/skip одного масштаба.
        Ожидает, что x и skip уже одной spatial-формы.
        Возвращает out того же spatial-размера, что и x.
        """
        B, _, H, W = x.size()

        q = self.query_conv(x).view(B, -1, H * W)   # (B, Cq, HW)
        k = self.key_conv(skip).view(B, -1, H * W)  # (B, Ck, HW)
        v = self.value_conv(skip).view(B, -1, H * W)  # (B, Cv, HW)

        qf = q.float()
        kf = k.float()
        vf = v.float()

        scale = max(1.0, kf.size(1)) ** 0.5
        sim = torch.bmm(qf.transpose(1, 2), kf) / scale   # (B, HW, HW)
        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = torch.softmax(sim, dim=-1)

        out = torch.bmm(vf, attn.transpose(1, 2)).view(B, -1, H, W)
        return out.to(x.dtype)

    def forward(self, x, skip):
        B, _, H, W = x.size()

        # приводим skip к тому же spatial, что и x
        if skip.shape[-2:] != (H, W):
            skip = F.interpolate(skip, size=(H, W), mode="bilinear", align_corners=False)

        hw = H * W

        if hw <= self.max_tokens:
            # карту можно обрабатывать напрямую
            out = self._attend(x, skip)
        else:
            # слишком большая: уменьшаем по пространству
            # подбираем шаг так, чтобы (H_s * W_s) <= max_tokens
            spatial_scale = int(math.ceil(math.sqrt(hw / float(self.max_tokens))))
            # защита от нуля
            spatial_scale = max(1, spatial_scale)

            x_small   = F.avg_pool2d(x,    kernel_size=spatial_scale,
                                     stride=spatial_scale, ceil_mode=True)
            skip_small = F.avg_pool2d(skip, kernel_size=spatial_scale,
                                      stride=spatial_scale, ceil_mode=True)

            out_small = self._attend(x_small, skip_small)
            out = F.interpolate(out_small, size=(H, W),
                                mode="bilinear", align_corners=False)

        return torch.cat([skip, self.gamma * out], dim=1)

class MAnetBlock(nn.Module):
    def __init__(self, out_channels, use_attention: bool, max_attn_tokens: int = 4096):
        super().__init__()
        self.use_attention = use_attention
        self.attention = PositionAttention(max_tokens=max_attn_tokens) if use_attention else None

        self.conv = nn.Sequential(
            nn.LazyConv2d(out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        # Стратегия U-Net: x апсемплим до размера skip, потом fusion.
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                                  align_corners=False)

            if self.use_attention and self.attention is not None:
                # глобальное внимание (с безопасным даунсемплингом внутри)
                att = self.attention(x, skip)   # вернёт concat(skip, gamma * out)
                x = torch.cat([x, att], dim=1)
            else:
                # дёшево: просто concat
                x = torch.cat([x, skip], dim=1)

        return self.conv(x)

class ModifiedMAnetDecoder(nn.Module):
    def __init__(
        self,
        decoder_channels,
        num_classes,
        use_attn_levels: int = 2,      # на скольких первых блоках включать attention
        max_attn_tokens: int = 4096,   # лимит на H*W внутри attention
    ):
        super().__init__()

        # features после feats[::-1]: [самый low-res, ..., самый high-res]
        # включаем внимание только на первых use_attn_levels (low-res),
        # там карты маленькие → память дёшево.
        self.blocks = nn.ModuleList([
            MAnetBlock(
                out_channels=ch,
                use_attention=(i < use_attn_levels),
                max_attn_tokens=max_attn_tokens
            )
            for i, ch in enumerate(decoder_channels)
        ])

        self.pre_final = nn.Sequential(
            nn.LazyConv2d(32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, features):
        """
        features: список после feats[::-1]:
        [low_res, ..., high_res], длина, как правило, 5.
        decoder_channels, например, [256, 128, 64, 32] → 4 блока.
        
        Returns: logits, feats, intermediate_feats
        intermediate_feats: list of decoder block outputs (for multi-scale HM)
        """
        x = features[0]  # самый глубоко-сжатый фичемап
        intermediate = []

        for i, block in enumerate(self.blocks):
            skip = features[i + 1] if (i + 1) < len(features) else None
            x = block(x, skip)  # внутри блока x апсемплится до размера skip
            intermediate.append(x)

        # x сейчас имеет spatial размер примерно как самый high-res skip
        feats = self.pre_final(x)
        feats = self.refine(feats)
        logits = self.head(feats)
        if torch.isnan(logits).any():
            print("NaN detected in seg_logits!")
        return logits, feats, intermediate

# V7: Seg-Point Cross-Attention
class SegPointCrossAttention(nn.Module):
    """Cross-attention: point features attend to seg features.
    Query = point_feat, Key/Value = seg_feat.
    Помогает точкам «видеть» контур сегментации."""
    def __init__(self, channels=256, num_heads=4, max_tokens=4096):
        super().__init__()
        self.max_tokens = max_tokens
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        self.gamma = nn.Parameter(torch.zeros(1))  # начинаем с нулевого вклада

    def forward(self, point_feat, seg_feat):
        B, C, H, W = point_feat.shape
        hw = H * W
        
        # Приводим seg_feat к тому же пространственному размеру
        if seg_feat.shape[-2:] != (H, W):
            seg_feat = F.interpolate(seg_feat, size=(H, W), mode='bilinear', align_corners=False)
        
        # Если слишком много токенов — даунсемплируем
        if hw > self.max_tokens:
            scale = int(math.ceil(math.sqrt(hw / float(self.max_tokens))))
            pf = F.avg_pool2d(point_feat, kernel_size=scale, stride=scale, ceil_mode=True)
            sf = F.avg_pool2d(seg_feat, kernel_size=scale, stride=scale, ceil_mode=True)
            out = self._attend(pf, sf)
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        else:
            out = self._attend(point_feat, seg_feat)
        
        return point_feat + self.gamma * out  # residual с learnable gate
    
    def _attend(self, point_feat, seg_feat):
        B, C, H, W = point_feat.shape
        q = point_feat.flatten(2).transpose(1, 2)   # (B, HW, C)
        kv = seg_feat.flatten(2).transpose(1, 2)    # (B, HW, C)
        
        q = self.norm_q(q)
        kv = self.norm_kv(kv)
        
        out, _ = self.attn(q, kv, kv)
        out = out + q  # внутренний residual
        out = self.ffn(out) + out
        
        return out.transpose(1, 2).view(B, C, H, W)


class DoublePointDetectionHead(nn.Module):
    def __init__(
        self,
        hidden_channels=256,
        num_points=4,
        use_coordinate_attention: bool = True,
        guidance_enabled: bool = False,
        offsets_enabled: bool = True,
    ):
        super().__init__()
        self.num_points = num_points
        self.guidance_enabled = guidance_enabled
        self.offsets_enabled = offsets_enabled
        self.point_conv = nn.Sequential(
            nn.LazyConv2d(hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        # v2: двойной Coordinate Attention каскадом
        self.ca = nn.Sequential(
            CoordinateAttention(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            CoordinateAttention(hidden_channels),
        ) if use_coordinate_attention else nn.Identity()
        
        # V7: Seg-Point Cross-Attention (point features attend to seg features)
        self.seg_cross_attn = SegPointCrossAttention(hidden_channels, num_heads=4)
        # V7: Projection для seg_feat (может иметь другое число каналов)
        self.seg_proj = nn.LazyConv2d(hidden_channels, 1)
        
        self.heatmap_conv = nn.Conv2d(hidden_channels, num_points, 1)
        # v2: улучшенный Offset Head
        self.offset_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, num_points * 2, 1),
        ) if offsets_enabled else None

    def forward(self, x, seg_feat=None, guidance_weight: float = 0.0):
        f = self.point_conv(x)
        f = self.ca(f)
        
        # V7: Cross-Attention с сегментационными фичами
        if seg_feat is not None:
            seg_proj = self.seg_proj(seg_feat)
            f = self.seg_cross_attn(f, seg_proj)

        heat = self.heatmap_conv(f)
        if self.offsets_enabled and self.offset_conv is not None:
            offsets = torch.sigmoid(self.offset_conv(f))
        else:
            offsets = torch.full(
                (f.shape[0], self.num_points * 2, f.shape[2], f.shape[3]),
                0.5,
                device=f.device,
                dtype=f.dtype,
            )
        return heat, None, offsets

class BoundaryAwareMAnet(nn.Module):
    def __init__(
        self,
        encoder_name: str = 'efficientnet_v2_s',
        num_classes: int = 1,
        num_points: int = 4,
        use_coordinate_attention: bool = True,
        decoder_attn_levels: int = 2,
        fusion_enabled: bool = True,
        point_head_channels: int = 256,   # каналов после fusion для point head
        max_attn_tokens: int = 4096,      # лимит для attention
        guidance_enabled: bool = True,
        offsets_enabled: bool = True,
    ):
        super().__init__()

        self.fusion_enabled = fusion_enabled
        self.encoder = safe_efficientnetv2(encoder_name)

        self.decoder = ModifiedMAnetDecoder(
            decoder_channels=[256, 128, 64, 32],
            num_classes=num_classes,
            use_attn_levels=max(0, int(decoder_attn_levels)),
            max_attn_tokens=max_attn_tokens,
        )

        # В обоих режимах приводим фичи к одинаковому числу каналов для головы точек.
        self.point_input_proj = nn.LazyConv2d(point_head_channels, 3, padding=1)

        self.point_head = DoublePointDetectionHead(
            hidden_channels=point_head_channels,
            num_points=num_points,
            use_coordinate_attention=use_coordinate_attention,
            guidance_enabled=guidance_enabled,
            offsets_enabled=offsets_enabled,
        )

        # v3 Proposal A: Отдельная голова для границ от seg_feats (decoupling)
        self.bnd_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

        # V7: Multi-Scale Heatmap heads
        # decoder_channels=[256, 128, 64, 32], берём последние 2 (64, 32)
        self.ms_heatmap_heads = nn.ModuleList([
            nn.Conv2d(64, num_points, 1),    # scale 1 (из decoder block 2, 64ch)
            nn.Conv2d(32, num_points, 1),    # scale 2 (из decoder block 3, 32ch)
        ])

    def forward(self, x, guidance_weight: float = 0.0):
        # --- Encoder ---
        feats = []
        h = x
        keep_idx = {1, 3, 5, 7, 9}
        for i, layer in enumerate(self.encoder):
            h = layer(h)
            if i in keep_idx:
                feats.append(h)

        feats = feats[::-1]

        # --- Decoder (V7: теперь возвращает промежуточные фичи) ---
        seg_logits, seg_feats, dec_intermediates = self.decoder(feats)

        if self.fusion_enabled:
            deep = F.interpolate(feats[0], size=seg_feats.shape[-2:], mode="bilinear", align_corners=False)
            point_input = torch.cat([deep, seg_feats], dim=1)
        else:
            point_input = seg_feats
        fused = self.point_input_proj(point_input)

        # --- Голова точек и границы ---
        # V7: передаём seg_feats в point_head для cross-attention
        heatmaps, _, offsets = self.point_head(fused, seg_feat=seg_feats, guidance_weight=guidance_weight)
        boundary = self.bnd_head(seg_feats)

        # V7: Multi-Scale Heatmaps из промежуточных фичей декодера
        # dec_intermediates: [256ch, 128ch, 64ch, 32ch]
        # Берём последние 2: 64ch (idx=2) и 32ch (idx=3)
        ms_heatmaps = []
        for i, head in enumerate(self.ms_heatmap_heads):
            feat_idx = i + 2  # индексы 2 и 3 в dec_intermediates
            if feat_idx < len(dec_intermediates):
                ms_hm = head(dec_intermediates[feat_idx])
                ms_heatmaps.append(ms_hm)

        return {
            "segmentation": seg_logits,
            "point_heatmaps": heatmaps,
            "boundary_weights": boundary,
            "point_offsets": offsets,
            "fused_features": fused,
            "stride": x.shape[-1] // seg_logits.shape[-1],
            "ms_heatmaps": ms_heatmaps,   # V7: multi-scale heatmaps
        }   

# -------------------------
# Differentiable spatial argmax
# -------------------------

def improved_softargmax2d(logits, beta: float = 10.0, stable: bool = True):
    logits = logits.float()

    B, P, H, W = logits.shape
    if stable:
        logits = logits - logits.amax(dim=(-1, -2), keepdim=True)

    # защита от overflow при больших beta / AMP
    logits = logits.clamp(-15.0, 15.0) 

    flat = (logits * float(beta)).view(B, P, -1)
    probs = torch.softmax(flat, dim=-1)

    yy, xx = torch.meshgrid(
        torch.arange(H, device=logits.device, dtype=torch.float32),
        torch.arange(W, device=logits.device, dtype=torch.float32),
        indexing='ij'
    )

    coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=0)  # (2, HW)
    coords = coords.view(1, 1, 2, H * W).expand(B, P, -1, -1)
    probs = probs.view(B, P, 1, H * W)

    exp = (probs * coords).sum(-1)
    return exp[:, :, 0], exp[:, :, 1]

# -------------------------
# Target builders
# -------------------------

def make_gaussian_heatmaps(points_vec, H, W, num_points=4, sigma=2.0, device=None):
    B = points_vec.shape[0]
    device = device or points_vec.device
    heat = torch.zeros((B, num_points, H, W), device=device)

    y = torch.arange(H, device=device).float().view(1, 1, H, 1)
    x = torch.arange(W, device=device).float().view(1, 1, 1, W)

    # v3: sigma can be a scalar or a tensor of shape (num_points,)
    if not isinstance(sigma, torch.Tensor):
        sigma = torch.full((num_points,), float(sigma), device=device)

    for j in range(num_points):
        px = points_vec[:, 3*j + 0].clamp(0, 1) * (W - 1)
        py = points_vec[:, 3*j + 1].clamp(0, 1) * (H - 1)
        v  = (points_vec[:, 3*j + 2] > 0.5).float()

        px = px.view(B, 1, 1, 1)
        py = py.view(B, 1, 1, 1)
        sj = sigma[j].view(1, 1, 1, 1)

        g = torch.exp(-((x - px)**2 + (y - py)**2) / (2 * sj * sj))
        g = g.view(B, 1, H, W)
        heat[:, j, :, :] = g[:, 0, :, :] * v.view(B, 1, 1)
    return heat.clamp(0, 1)

# def sobel_boundary_targets(mask, down_h, down_w):
#     with torch.no_grad():
#         m = F.interpolate(mask, size=(down_h, down_w), mode="bilinear", align_corners=False)
#         sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=m.dtype, device=m.device).view(1,1,3,3)
#         sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=m.dtype, device=m.device).view(1,1,3,3)
#         gx = F.conv2d(m, sobel_x, padding=1)
#         gy = F.conv2d(m, sobel_y, padding=1)
#         mag = torch.sqrt(gx*gx + gy*gy)
#         mag = mag / (mag.amax(dim=(2,3), keepdim=True) + 1e-6)
#         return mag.clamp(0, 1)

def sobel_boundary_targets(mask, down_h, down_w, smooth_iters: int = 2):

    with torch.no_grad():
        # 1. Даунсемплинг GT-маски (как у тебя было, чтобы не ломать геометрию)
        m = F.interpolate(
            mask, size=(down_h, down_w),
            mode="bilinear", align_corners=False
        )

        # 2. Лёгкое сглаживание маски
        #    Можно увеличить smooth_iters до 3, если хочешь более гладкую границу
        for _ in range(smooth_iters):
            m = F.avg_pool2d(m, kernel_size=3, stride=1, padding=1)

        # 3. Собель как раньше
        sobel_x = torch.tensor(
            [[1, 0, -1],
             [2, 0, -2],
             [1, 0, -1]],
            dtype=m.dtype, device=m.device
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[1,  2,  1],
             [0,  0,  0],
             [-1, -2, -1]],
            dtype=m.dtype, device=m.device
        ).view(1, 1, 3, 3)

        gx = F.conv2d(m, sobel_x, padding=1)
        gy = F.conv2d(m, sobel_y, padding=1)

        mag = torch.sqrt(gx * gx + gy * gy)

        # Нормализуем, но защищаемся от деления на почти ноль
        denom = mag.amax(dim=(2, 3), keepdim=True).clamp_min(1e-3)
        mag = mag / denom

        return mag.clamp(0.0, 1.0)

# -------------------------
# Lightning system
# -------------------------

class LitBoundaryAwareSystem(pl.LightningModule):
    def __init__(self,
                 encoder_name: str = 'efficientnet_v2_s',
                 num_classes: int = 1,
                 num_points: int = 4,
                 warmup_epochs: int = 5,
                 lr: float = 1e-3,
                 weight_seg_bce: float = 1.0,
                 weight_seg_dice: float = 1.0,
                 weight_pts: float = 1.0,
                 weight_bnd: float = 0.5,
                 point_sigma: float = 2.0,
                 use_focal_seg: bool = True,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 use_improved_points: bool = True,
                 ip_alpha: float = 0.25,
                 ip_beta: float = 0.1,
                 ip_gamma: float = 2.0,
                 use_coordinate_attention: bool = True,
                 decoder_attn_levels: int = 2,
                 fusion_enabled: bool = True,
                 point_head_channels: int = 256,
                 max_attn_tokens: int = 4096,
                 boundary_guidance_enabled: bool = True,
                 boundary_loss_enabled: bool = True,
                 offsets_enabled: bool = True,
                 softargmax_beta_mode: str = "schedule",
                 softargmax_beta_fixed: float = 10.0,
                 softargmax_beta_start: float = 4.0,
                 softargmax_beta_end: float = 12.0,
                 softargmax_beta_warmup_epochs: int = 8):
        super().__init__()
        self.save_hyperparameters()

        self.model = BoundaryAwareMAnet(
            encoder_name=encoder_name,
            num_classes=num_classes,
            num_points=num_points,
            use_coordinate_attention=use_coordinate_attention,
            decoder_attn_levels=decoder_attn_levels,
            fusion_enabled=fusion_enabled,
            point_head_channels=point_head_channels,
            max_attn_tokens=max_attn_tokens,
            guidance_enabled=boundary_guidance_enabled,
            offsets_enabled=offsets_enabled,
        )

        self.seg_criterion = (FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')
                              if use_focal_seg else nn.BCEWithLogitsLoss())
        #self.bce = nn.BCEWithLogitsLoss()
        self.register_buffer("bce_pos_weight", torch.tensor([2.0], dtype=torch.float32))
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.bce_pos_weight)
        self.use_improved_points = use_improved_points
        self.improved_point_loss = ImprovedPointLoss(alpha=ip_alpha, beta=ip_beta, gamma=ip_gamma)

        # v3 Proposal B: Learnable Loss Weights (Uncertainty Weighting)
        self.log_var_seg = nn.Parameter(torch.zeros(1))
        self.log_var_pts = nn.Parameter(torch.zeros(1))
        self.log_var_bnd = nn.Parameter(torch.zeros(1))

        # v3 Proposal C: Learnable sigma per point
        # Initial sigma from point_sigma param
        init_sigma = float(point_sigma)
        self.log_sigma = nn.Parameter(torch.full((num_points,), math.log(1.5)))

        self.max_val_saves = 5
        
        # Списки для сбора метрик на каждой эпохе
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def _current_softargmax_beta(self, epoch: int) -> float:
        beta_mode = str(getattr(self.hparams, "softargmax_beta_mode", "schedule")).strip().lower()
        if beta_mode == "fixed":
            return float(getattr(self.hparams, "softargmax_beta_fixed", 10.0))

        beta_start = float(getattr(self.hparams, "softargmax_beta_start", 4.0))
        beta_end = float(getattr(self.hparams, "softargmax_beta_end", 16.0))        # v2: 12→16
        beta_warm = max(1, int(getattr(self.hparams, "softargmax_beta_warmup_epochs", 12)))  # v2: 8→12
        progress = min(1.0, max(0.0, epoch / float(beta_warm)))
        return beta_start + (beta_end - beta_start) * progress

    def configure_optimizers(self):
        sigma_params = [self.log_sigma]
        other_params = [p for n, p in self.named_parameters() if 'log_sigma' not in n]
        
        optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': self.hparams.lr},
            {'params': sigma_params, 'lr': self.hparams.lr * 3},  # 3x LR (V5) вместо 10x
        ], weight_decay=3e-4)
        # V6: Адаптивный LR (ReduceLROnPlateau)
        # Уменьшаем LR в 2 раза, если val_balance_score не растет 5 эпох
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-6
        )
 
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler, 
                "monitor": "val_balance_score", 
                "interval": "epoch"
            }
        }

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, stage: str):
        x, targets = batch
        mask, points = targets["mask"], targets["points"]
        points = torch.nan_to_num(points, nan=0.0)
        mask = torch.nan_to_num(mask, nan=0.0)
        epoch = int(self.current_epoch)
        beta = self._current_softargmax_beta(epoch)
        
        # v3 Proposal C: Learnable Sigma
        # v5: ужесточаем clamp до 2.5 (было 5.0)
        sigma = torch.exp(self.log_sigma).clamp(0.5, 2.5)
        
        # v3 Proposal B: Uncertainty Weights
        w_seg_bce = float(self.hparams.weight_seg_bce)
        w_seg_dice = float(self.hparams.weight_seg_dice)
        w_pts_init = float(self.hparams.weight_pts)
        w_bnd_init = float(self.hparams.weight_bnd)

        # v3: boundary loss и guidance снова включены, но безопасно!
        boundary_loss_enabled = bool(getattr(self.hparams, "boundary_loss_enabled", True))
        guidance_enabled = bool(getattr(self.hparams, "boundary_guidance_enabled", True))
        offsets_enabled = bool(getattr(self.hparams, "offsets_enabled", True))

        guidance_weight = 0.0
        offset_gain = max(0.5, min(1.0, (epoch - 12.0) / 10.0)) if offsets_enabled else 0.0

        # ----- forward -----
        out = self.model(x, guidance_weight=guidance_weight)
        seg_logits = out["segmentation"]
        heat_pred  = out["point_heatmaps"]
        bnd_pred   = out["boundary_weights"]
        offsets_pred = out["point_offsets"]

        h, w = seg_logits.shape[-2:]
        mask_down = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)

        # цели для точек
        heat_tgt = make_gaussian_heatmaps(
            points, h, w,
            num_points=self.hparams.num_points,
            sigma=sigma,
            device=mask.device
        )

        bnd_tgt = sobel_boundary_targets(mask, h, w, smooth_iters=2)

        # ----- лоссы сегментации -----
        loss_seg_main = self.seg_criterion(seg_logits, mask_down)
        loss_dice     = dice_loss(seg_logits, mask_down)

        # ----- координаты точек -----
        px_hm, py_hm = improved_softargmax2d(heat_pred, beta=beta, stable=True)
        
        B, P, H, W = heat_pred.shape
        
        grid_x = (px_hm / max(1.0, float(W - 1))) * 2.0 - 1.0
        grid_y = (py_hm / max(1.0, float(H - 1))) * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2).clamp(-1.0, 1.0) 
        
        offset_x_map = offsets_pred[:, 0::2, :, :] # (B, P, H, W)
        offset_y_map = offsets_pred[:, 1::2, :, :] # (B, P, H, W)
        
        # grid_sample выдаст (B, Channels, Points, 1). После squeeze(-1) -> (B, P, P)
        sample_x_all = F.grid_sample(offset_x_map, grid, mode='bilinear', align_corners=True).squeeze(-1)
        sample_y_all = F.grid_sample(offset_y_map, grid, mode='bilinear', align_corners=True).squeeze(-1)
        
        # Нам нужно значение для i-й точки ИЗ i-го канала (берем диагональ по размерностям 1 и 2)
        # Получаем итоговый размер (B, P)
        sample_offset_x = torch.diagonal(sample_x_all, dim1=1, dim2=2)
        sample_offset_y = torch.diagonal(sample_y_all, dim1=1, dim2=2)
        
        # Финальные координаты = Heatmap Coords + Offset Correction
        # Поскольку в голове была sigmoid, значения лежат в [0, 1].
        # Приводим их к диапазону [-1.0, 1.0] пикселя, чтобы модель могла двигать точку в любую сторону!
        sample_offset_x = (sample_offset_x - 0.5) * 2.0 * offset_gain
        sample_offset_y = (sample_offset_y - 0.5) * 2.0 * offset_gain
        
        final_px = px_hm + sample_offset_x
        final_py = py_hm + sample_offset_y
        
        coords_pred = torch.stack([final_px, final_py], dim=-1)

        B, P, W0, H0 = points.shape[0], self.hparams.num_points, w - 1, h - 1
        coords_tgt = torch.zeros((B, P, 2), device=heat_pred.device, dtype=heat_pred.dtype)
        vis        = torch.zeros((B, P),   device=heat_pred.device, dtype=heat_pred.dtype)

        for j in range(P):
            coords_tgt[:, j, 0] = points[:, 3*j + 0].clamp(0, 1) * W0
            coords_tgt[:, j, 1] = points[:, 3*j + 1].clamp(0, 1) * H0
            vis[:, j]          = (points[:, 3*j + 2] > 0.5).float()

        if self.use_improved_points:
            loss_pts_total, loss_pts_hm, loss_pts_coord = self.improved_point_loss(
                heat_pred, heat_tgt, coords_pred, coords_tgt, vis
            )
        else:
            loss_pts_hm    = F.binary_cross_entropy_with_logits(heat_pred, heat_tgt)
            loss_pts_coord = torch.zeros(1, device=x.device, dtype=heat_pred.dtype)
            loss_pts_total = loss_pts_hm

        # V7: Multi-Scale Heatmap Loss
        ms_heatmaps = out.get("ms_heatmaps", [])
        loss_ms_hm = torch.zeros((), device=x.device, dtype=heat_pred.dtype)
        if ms_heatmaps:
            for ms_hm in ms_heatmaps:
                # Апсемплим таргет до размера промежуточной heatmap
                ms_h, ms_w = ms_hm.shape[-2:]
                ms_tgt = F.interpolate(heat_tgt, size=(ms_h, ms_w), mode='bilinear', align_corners=False)
                loss_ms_hm += F.binary_cross_entropy_with_logits(ms_hm, ms_tgt)
            
            # Добавляем MS лосс к общему лоссу точек с небольшим весом
            loss_pts_total = loss_pts_total + 0.5 * loss_ms_hm

        # ----- boundary-loss: v3 — СНОВА ВКЛЮЧЁН (через отдельную голову) -----
        if boundary_loss_enabled:
            bnd_pred_sig = torch.sigmoid(bnd_pred)
            loss_bnd = F.smooth_l1_loss(bnd_pred_sig, bnd_tgt)
        else:
            loss_bnd = torch.zeros((), device=x.device, dtype=heat_pred.dtype)

        # ----- v3: Uncertainty Weighting (Proposal B) -----
        # Считаем взвешенные лоссы с ограничением (clamp) для стабильности
        lv_seg = self.log_var_seg.clamp(-2.0, 2.0)
        lv_pts = self.log_var_pts.clamp(-2.0, 2.0)
        lv_bnd = self.log_var_bnd.clamp(-2.0, 2.0)

        precision_seg = torch.exp(-lv_seg)
        precision_pts = torch.exp(-lv_pts)
        precision_bnd = torch.exp(-lv_bnd)

        loss_seg_combined = w_seg_bce * loss_seg_main + w_seg_dice * loss_dice
        
        # Основной лосс по Kendall et al.
        # v5: используем exp(lv) вместо lv, чтобы лосс был всегда положительным
        loss_main = (precision_seg * loss_seg_combined + torch.exp(lv_seg)) + \
                    (precision_pts * (w_pts_init * loss_pts_total) + torch.exp(lv_pts)) + \
                    (precision_bnd * (w_bnd_init * loss_bnd) + torch.exp(lv_bnd))
        
        # V4: L2-регуляризация на log_var для предотвращения дрейфа весов
        loss_reg = 0.01 * (self.log_var_seg**2 + self.log_var_pts**2 + self.log_var_bnd**2)
        
        # V5: Штраф за слишком большую sigma (подталкиваем к 1.5)
        loss_sigma_penalty = 0.1 * (sigma - 1.5).clamp(min=0).mean()
        
        loss = loss_main + loss_reg + loss_sigma_penalty

        d = dice_coef(seg_logits, mask_down)

        output = {
            "loss": loss, "dice": d,
            "loss_seg": loss_seg_main, "loss_dice": loss_dice,
            "loss_pts": loss_pts_total, "loss_pts_hm": loss_pts_hm, "loss_pts_coord": loss_pts_coord,
            "loss_bnd": loss_bnd,
            "w_pts_eff": precision_pts.detach(),
            "w_bnd_eff": precision_bnd.detach(),
            "w_seg_eff": precision_seg.detach(),
            "sigma_min": sigma.min().detach(),
            "sigma_max": sigma.max().detach(),
            "offset_gain_eff": torch.tensor(offset_gain, device=self.device, dtype=torch.float32),
        }

        if stage == "val":
            with torch.no_grad():
                B, P, hhm, whm = heat_pred.shape
                H0f, W0f = mask.shape[-2:]

                # 1) Базовые координаты из heatmap
                px_hm, py_hm = improved_softargmax2d(
                    heat_pred, beta=beta, stable=True
                )  # (B, P)

                # 2) Уточнение координат через offsets_pred
                grid_x = (px_hm / max(1.0, float(whm - 1))) * 2.0 - 1.0
                grid_y = (py_hm / max(1.0, float(hhm - 1))) * 2.0 - 1.0
                grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2).clamp(-1.0, 1.0)  # (B, P, 1, 2)

                offset_x_map = offsets_pred[:, 0::2, :, :]  # (B, P, H, W)
                offset_y_map = offsets_pred[:, 1::2, :, :]  # (B, P, H, W)

                sample_x_all = F.grid_sample(
                    offset_x_map, grid, mode='bilinear', align_corners=True
                ).squeeze(-1)  # (B, P, P)

                sample_y_all = F.grid_sample(
                    offset_y_map, grid, mode='bilinear', align_corners=True
                ).squeeze(-1)  # (B, P, P)

                # Берём диагональ: i-я точка из i-го канала
                sample_offset_x = torch.diagonal(sample_x_all, dim1=1, dim2=2)  # (B, P)
                sample_offset_y = torch.diagonal(sample_y_all, dim1=1, dim2=2)  # (B, P)

                # sigmoid -> [0,1], переводим в [-1,1] пикселя
                sample_offset_x = (sample_offset_x - 0.5) * 2.0
                sample_offset_y = (sample_offset_y - 0.5) * 2.0

                # Финальные координаты в пространстве heatmap
                final_px = px_hm + sample_offset_x
                final_py = py_hm + sample_offset_y

                # 3) Переводим финальные координаты в размер исходной маски
                scale_x = (W0f - 1) / max(1.0, float(whm - 1))
                scale_y = (H0f - 1) / max(1.0, float(hhm - 1))
                pred_x = final_px * scale_x
                pred_y = final_py * scale_y

                # 4) GT-координаты
                gt_x = torch.zeros_like(pred_x)
                gt_y = torch.zeros_like(pred_y)
                vism = torch.zeros_like(pred_x)

                for j in range(self.hparams.num_points):
                    gt_x[:, j] = points[:, 3*j + 0].clamp(0, 1) * (W0f - 1)
                    gt_y[:, j] = points[:, 3*j + 1].clamp(0, 1) * (H0f - 1)

                    # лучше ориентироваться в первую очередь на visibility-бит
                    vism[:, j] = (points[:, 3*j + 2] > 0.5).float()

                # 5) Ошибка только по видимым точкам
                err = torch.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
                vis_mask = (vism > 0.5).float()

                sum_err = (err * vis_mask).sum()
                vis_count = int(vis_mask.sum().item())

                if vis_count > 0:
                    self._val_err_sum += float(sum_err.item())
                    self._val_vis_cnt += vis_count

                self.log("val/pt_vis_count",
                         torch.tensor((vis_count / float(vis_mask.numel())) if vis_mask.numel() else 0.0,
                                      device=x.device),
                         prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0))

                save_max = getattr(self, "max_val_saves", 5)
                save_cnt = getattr(self, "val_save_count", 0)
                if save_cnt < save_max:
                    log_dir = None
                    if self.logger is not None:
                        if hasattr(self.logger, "log_dir") and self.logger.log_dir:
                            log_dir = self.logger.log_dir
                        elif hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "log_dir"):
                            log_dir = self.logger.experiment.log_dir
                    if log_dir is None:
                        log_dir = self.trainer.default_root_dir
                    out_dir = os.path.join(log_dir, "val_vis")
                    os.makedirs(out_dir, exist_ok=True)

                    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

                    seg_prob = torch.sigmoid(seg_logits)
                    seg_up = torch.nn.functional.interpolate(seg_prob, size=(H0f, W0f), mode="bilinear", align_corners=False)
                    img_up = torch.nn.functional.interpolate(x, size=(H0f, W0f), mode="bilinear", align_corners=False)

                    for b in range(min(x.size(0), save_max - save_cnt)):
                        fig = plt.figure(figsize=(6, 6))

                        img = img_up[b].detach().cpu()
                        img_np = img.permute(1, 2, 0).numpy()
                        img_np = np.clip(img_np * IMAGENET_STD + IMAGENET_MEAN, 0.0, 1.0)

                        img_gray = img_np.mean(axis=2)
                        plt.imshow(img_gray, cmap="gray", vmin=0.0, vmax=1.0)

                        seg_np = seg_up[b, 0].detach().cpu().numpy()
                        plt.imshow(seg_np, alpha=0.35)

                        for j in range(P):
                            if vism[b, j] > 0.5:
                                plt.scatter([gt_x[b, j].item()], [gt_y[b, j].item()], s=30, marker='o')
                            plt.scatter([pred_x[b, j].item()], [pred_y[b, j].item()], s=30, marker='x')

                        plt.axis('off')
                        fname = os.path.join(out_dir, f"ep{self.current_epoch:03d}_b{self.global_step:06d}_{save_cnt+b}.png")
                        plt.tight_layout(pad=0)
                        plt.savefig(fname, dpi=150, bbox_inches='tight')
                        plt.close(fig)

                    self.val_save_count = save_cnt + min(x.size(0), save_max - save_cnt)
        return output

    def training_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, "train")
        detached_outputs = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}
        self.training_step_outputs.append(detached_outputs)

        self.log_dict(
            {
                'loss': outputs['loss'].detach(),
                'dice': outputs['dice'].detach()
            },
            on_step=True, on_epoch=False, prog_bar=True, logger=False
        )
        return outputs['loss']

    def validation_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, "val")
        detached_outputs = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}
        self.validation_step_outputs.append(detached_outputs)

        self.log_dict(
            {
                'val_loss': outputs['loss'].detach(),
                'val_dice': outputs['dice'].detach()
            },
            on_step=True, on_epoch=False, prog_bar=True, logger=False
        )

    def on_validation_epoch_start(self):
        # Инициализация переменных перед каждой эпохой валидации
        self.val_save_count = 0
        self._val_err_sum = 0.0
        self._val_vis_cnt = 0

    def on_validation_epoch_end(self):
        mean_err_epoch = self._val_err_sum / self._val_vis_cnt if self._val_vis_cnt > 0 else 0.0

        if self.validation_step_outputs:
            val_dice_epoch = torch.stack([x["dice"] for x in self.validation_step_outputs]).mean()
        else:
            val_dice_epoch = torch.tensor(0.0, device=self.device)

        # point score: чем меньше ошибка, тем выше score
        pt_score = 1.0 / (1.0 + (mean_err_epoch / 10.0))

        # баланс сегментации и точек
        balance_score = 0.6 * float(val_dice_epoch.item()) + 0.4 * float(pt_score)

        self.log("val_pt_err_px", torch.tensor(mean_err_epoch, device=self.device), prog_bar=True, on_epoch=True, logger=True)
        self.log("val_dice_epoch", val_dice_epoch.detach(), prog_bar=False, on_epoch=True, logger=True)
        self.log("val_balance_score", torch.tensor(balance_score, device=self.device), prog_bar=True, on_epoch=True, logger=True)

        train_avg_metrics = {}
        if self.training_step_outputs:
            for key in self.training_step_outputs[0].keys():
                vals = [x[key] for x in self.training_step_outputs]
                if all(torch.is_tensor(v) for v in vals):
                    stacked = torch.stack([v.float() for v in vals])
                    train_avg_metrics[f"train/{key}"] = stacked.mean()

        val_avg_metrics = {}
        if self.validation_step_outputs:
            for key in self.validation_step_outputs[0].keys():
                vals = [x[key] for x in self.validation_step_outputs]
                if all(torch.is_tensor(v) for v in vals):
                    stacked = torch.stack([v.float() for v in vals])
                    val_avg_metrics[f"val/{key}"] = stacked.mean()

        all_metrics = {**train_avg_metrics, **val_avg_metrics}
        if all_metrics:
            self.log_dict(all_metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.training_step_outputs.clear()
        self.validation_step_outputs.clear()

def build_system(cfg: dict):
    enc = cfg.get("encoder_name", "efficientnet_v2_s")
    num_classes = int(cfg.get("num_classes", 1))
    num_points  = int(cfg.get("num_points", 4))
    warmup_epochs = int(cfg.get("warmup_epochs", 5))
    lr          = float(cfg.get("lr", 1e-3))

    loss_cfg = cfg.get("loss", {})
    focal_cfg = loss_cfg.get("focal", {})
    ip_cfg = loss_cfg.get("improved_points", {})
    attn_cfg = cfg.get("attention", {})
    fusion_cfg = cfg.get("fusion", {})
    boundary_cfg = cfg.get("boundary", {})
    offsets_cfg = cfg.get("offsets", {})
    softargmax_cfg = cfg.get("softargmax", {})
    legacy_loss_boundary_cfg = loss_cfg.get("boundary", {})

    decoder_enabled = bool(attn_cfg.get("decoder_enabled", attn_cfg.get("decoder", True)))
    decoder_attn_levels = int(attn_cfg.get("decoder_attn_levels", 2)) if decoder_enabled else 0
    boundary_loss_enabled = bool(boundary_cfg.get("loss_enabled", legacy_loss_boundary_cfg.get("enabled", True)))
    boundary_guidance_enabled = bool(boundary_cfg.get("guidance_enabled", True))

    return LitBoundaryAwareSystem(
        encoder_name=enc,
        num_classes=num_classes,
        num_points=num_points,
        warmup_epochs=warmup_epochs,
        lr=lr,
        weight_seg_bce=float(loss_cfg.get("w_bce", 1.0)),
        weight_seg_dice=float(loss_cfg.get("w_dice", 1.0)),
        weight_pts=float(loss_cfg.get("w_pts", 1.0)),
        weight_bnd=float(loss_cfg.get("w_bnd", 0.5)),
        point_sigma=float(loss_cfg.get("point_sigma", 2.0)),
        use_focal_seg=bool(focal_cfg.get("use_focal_seg", True)),
        focal_alpha=float(focal_cfg.get("alpha", 0.25)),
        focal_gamma=float(focal_cfg.get("gamma", 2.0)),
        use_improved_points=bool(ip_cfg.get("enabled", True)),
        ip_alpha=float(ip_cfg.get("alpha", 0.25)),
        ip_beta=float(ip_cfg.get("beta", 0.1)),
        ip_gamma=float(ip_cfg.get("gamma", 2.0)),
        use_coordinate_attention=bool(attn_cfg.get("coordinate", True)),
        decoder_attn_levels=decoder_attn_levels,
        fusion_enabled=bool(fusion_cfg.get("enabled", True)),
        point_head_channels=int(fusion_cfg.get("point_head_channels", 256)),
        max_attn_tokens=int(attn_cfg.get("max_attn_tokens", 4096)),
        boundary_guidance_enabled=boundary_guidance_enabled,
        boundary_loss_enabled=boundary_loss_enabled,
        offsets_enabled=bool(offsets_cfg.get("enabled", True)),
        softargmax_beta_mode=str(softargmax_cfg.get("beta_mode", "schedule")),
        softargmax_beta_fixed=float(softargmax_cfg.get("beta_fixed", 10.0)),
        softargmax_beta_start=float(softargmax_cfg.get("beta_start", 4.0)),
        softargmax_beta_end=float(softargmax_cfg.get("beta_end", 16.0)),
        softargmax_beta_warmup_epochs=int(softargmax_cfg.get("beta_warmup_epochs", 12)),
    )
