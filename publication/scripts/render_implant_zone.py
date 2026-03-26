from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torchvision import models as tv_models

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bamnet_paths import expand_config_tree, get_data_path
import model_backbone_coords as model_module
from model_backbone_coords import BoundaryAwareMAnet, improved_softargmax2d


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
POINT_NAMES = ["AA1", "AA2", "STJ1", "STJ2"]


def find_default_weights() -> Path | None:
    preferred = ROOT_DIR / "boundary_aware_manet_v4" / "20260323_003555" / "best.pt"
    if preferred.is_file():
        return preferred

    best_candidates = sorted(
        ROOT_DIR.glob("boundary_aware_manet_v4/*/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if best_candidates:
        return best_candidates[0]

    ckpt_candidates = sorted(
        ROOT_DIR.glob("boundary_aware_manet_v4/*/checkpoints/*.ckpt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if ckpt_candidates:
        return ckpt_candidates[0]

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference for aortic root mask/points and overlay implant zone."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=find_default_weights(),
        help="Path to a BAMNet Lightning checkpoint (.ckpt) or best.pt symlink/copy. "
             "If omitted, the script tries to auto-detect weights under boundary_aware_manet_v4/.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT_DIR / "config.yaml",
        help="Training config used to construct the model.",
    )
    parser.add_argument(
        "--input",
        "--images",
        dest="input_path",
        type=Path,
        default=ROOT_DIR / "publication" / "figures",
        help="Input image file or directory with images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "publication" / "rendered_implant_zone",
        help="Directory for rendered overlays.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device, for example auto, cpu or cuda:0.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Segmentation threshold after sigmoid.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Inference resize; defaults to config img_size or original image size.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return expand_config_tree(yaml.safe_load(fh))


def patch_backbone_loader() -> None:
    def _offline_safe_efficientnetv2(encoder_name: str):
        name = encoder_name.lower()
        if "efficientnet_v2_s" in name:
            return tv_models.efficientnet_v2_s(weights=None).features
        if "efficientnet_v2_m" in name:
            return tv_models.efficientnet_v2_m(weights=None).features
        return tv_models.efficientnet_v2_l(weights=None).features

    model_module.safe_efficientnetv2 = _offline_safe_efficientnetv2


def read_checkpoint_bytes(path: Path) -> bytes:
    with path.open("rb") as fh:
        return fh.read()


def load_checkpoint(path: Path, device: torch.device):
    raw = read_checkpoint_bytes(path)
    try:
        return torch.load(io.BytesIO(raw), map_location=device, weights_only=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load checkpoint '{path}'. "
            f"The file looks corrupted or incomplete: {exc}"
        ) from exc


def find_fallback_checkpoint(weights_path: Path) -> Path | None:
    if weights_path.suffix.lower() != ".pt":
        return None
    checkpoints_dir = weights_path.parent / "checkpoints"
    if not checkpoints_dir.exists():
        return None
    candidates = sorted(checkpoints_dir.glob("*.ckpt"))
    if not candidates:
        return None
    return candidates[0]


def extract_state_dict(obj) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
            value = obj.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    raise RuntimeError("Unsupported checkpoint structure: state dict not found.")


def strip_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = ("model.", "module.", "net.")
    stripped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        stripped[new_key] = value
    return stripped


def build_model(cfg: dict, device: torch.device) -> BoundaryAwareMAnet:
    patch_backbone_loader()
    encoder_name = cfg.get("encoder_name", "efficientnet_v2_m")
    num_classes = int(cfg.get("num_classes", 1))
    num_points = int(cfg.get("num_points", 4))
    attn_cfg = cfg.get("attention", {})
    fusion_cfg = cfg.get("fusion", {})
    boundary_cfg = cfg.get("boundary", {})
    offsets_cfg = cfg.get("offsets", {})
    decoder_enabled = bool(attn_cfg.get("decoder_enabled", attn_cfg.get("decoder", True)))
    decoder_attn_levels = int(attn_cfg.get("decoder_attn_levels", 2)) if decoder_enabled else 0
    model = BoundaryAwareMAnet(
        encoder_name=encoder_name,
        num_classes=num_classes,
        num_points=num_points,
        use_coordinate_attention=bool(attn_cfg.get("coordinate", True)),
        decoder_attn_levels=decoder_attn_levels,
        fusion_enabled=bool(fusion_cfg.get("enabled", True)),
        point_head_channels=int(fusion_cfg.get("point_head_channels", 256)),
        max_attn_tokens=int(attn_cfg.get("max_attn_tokens", 4096)),
        guidance_enabled=bool(boundary_cfg.get("guidance_enabled", True)),
        offsets_enabled=bool(offsets_cfg.get("enabled", True)),
    )
    return model.to(device).eval()


def prepare_image(image_bgr: np.ndarray, img_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    orig_h, orig_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    tensor = resized.astype(np.float32) / 255.0
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    tensor = np.transpose(tensor, (2, 0, 1))
    return torch.from_numpy(tensor).unsqueeze(0), (orig_h, orig_w)


def infer_softargmax_beta(cfg: dict) -> float:
    softargmax_cfg = cfg.get("softargmax", {})
    mode = str(softargmax_cfg.get("beta_mode", "schedule")).strip().lower()
    if mode == "fixed":
        return float(softargmax_cfg.get("beta_fixed", 10.0))
    return float(softargmax_cfg.get("beta_end", 12.0))


def predict_points(
    point_heatmaps: torch.Tensor,
    orig_size: Tuple[int, int],
    beta: float,
) -> np.ndarray:
    px_hm, py_hm = improved_softargmax2d(point_heatmaps, beta=beta, stable=True)
    bsz, num_points, height, width = point_heatmaps.shape
    orig_h, orig_w = orig_size
    final_px = px_hm * (orig_w / float(width))
    final_py = py_hm * (orig_h / float(height))
    coords = torch.stack([final_px, final_py], dim=-1)
    return coords[0].detach().cpu().numpy()


def mask_to_overlay(mask_binary: np.ndarray) -> np.ndarray:
    overlay = np.zeros((mask_binary.shape[0], mask_binary.shape[1], 3), dtype=np.uint8)
    overlay[..., 1] = 220
    overlay[..., 2] = 150
    overlay[mask_binary == 0] = 0
    return overlay


def normalize(vec: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / length).astype(np.float32)


def line_segment(center: np.ndarray, direction: np.ndarray, half_length: float) -> np.ndarray:
    delta = direction * half_length
    return np.stack([center - delta, center + delta], axis=0)


def cross2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def intersect_line_with_segment(
    line_point: np.ndarray,
    line_direction: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
) -> np.ndarray | None:
    seg_direction = seg_b - seg_a
    denom = cross2d(line_direction, seg_direction)
    if abs(denom) < 1e-6:
        return None

    delta = seg_a - line_point
    t = cross2d(delta, seg_direction) / denom
    u = cross2d(delta, line_direction) / denom

    if u < -1e-3 or u > 1.0 + 1e-3:
        return None
    return line_point + line_direction * t


def segment_between_root_edges(
    center: np.ndarray,
    line_direction: np.ndarray,
    left_edge: Tuple[np.ndarray, np.ndarray],
    right_edge: Tuple[np.ndarray, np.ndarray],
    fallback_half_length: float,
) -> np.ndarray:
    left_hit = intersect_line_with_segment(center, line_direction, left_edge[0], left_edge[1])
    right_hit = intersect_line_with_segment(center, line_direction, right_edge[0], right_edge[1])

    if left_hit is not None and right_hit is not None:
        return np.stack([left_hit, right_hit], axis=0)

    return line_segment(center, line_direction, fallback_half_length)


def segment_strip(segment: np.ndarray, normal_direction: np.ndarray, half_thickness: float) -> np.ndarray:
    normal_delta = normal_direction * half_thickness
    return np.stack(
        [
            segment[0] + normal_delta,
            segment[1] + normal_delta,
            segment[1] - normal_delta,
            segment[0] - normal_delta,
        ],
        axis=0,
    )


def as_int_points(points: np.ndarray) -> np.ndarray:
    return np.round(points).astype(np.int32).reshape((-1, 1, 2))


def draw_zone_geometry(canvas: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    aa1, aa2, stj1, stj2 = [points_xy[idx].astype(np.float32) for idx in range(4)]
    annulus_mid = (aa1 + aa2) * 0.5
    stj_mid = (stj1 + stj2) * 0.5

    ring_direction = normalize(aa2 - aa1)
    axis_seed = normalize(stj_mid - annulus_mid)
    axis_ortho = axis_seed - ring_direction * float(np.dot(axis_seed, ring_direction))
    if np.linalg.norm(axis_ortho) < 1e-6:
        axis_ortho = np.array([-ring_direction[1], ring_direction[0]], dtype=np.float32)
    root_axis_direction = normalize(axis_ortho)
    if float(np.dot(root_axis_direction, stj_mid - annulus_mid)) < 0.0:
        root_axis_direction = -root_axis_direction

    root_depth_px = max(1.0, float(np.dot(stj_mid - annulus_mid, root_axis_direction)))
    annulus_width_px = max(1.0, float(np.linalg.norm(aa2 - aa1)))
    stj_width_px = max(1.0, float(np.linalg.norm(stj2 - stj1)))
    fallback_half_length = 0.5 * max(annulus_width_px, stj_width_px)

    zone_start_px = root_depth_px * 0.18
    zone_end_px = root_depth_px * 0.82
    ci_half_width_px = max(4.0, root_depth_px * 0.08)
    axis_before_ring_px = root_depth_px * 0.35
    axis_beyond_stj_px = root_depth_px * 0.45

    left_edge = (aa1, stj1)
    right_edge = (aa2, stj2)

    root_polygon = np.stack([aa1, aa2, stj2, stj1], axis=0)
    zone_lower_center = annulus_mid + root_axis_direction * zone_start_px
    zone_upper_center = annulus_mid + root_axis_direction * zone_end_px
    zone_lower_seg = segment_between_root_edges(
        zone_lower_center,
        ring_direction,
        left_edge,
        right_edge,
        fallback_half_length,
    )
    zone_upper_seg = segment_between_root_edges(
        zone_upper_center,
        ring_direction,
        left_edge,
        right_edge,
        fallback_half_length,
    )
    zone_polygon = np.stack(
        [zone_lower_seg[0], zone_lower_seg[1], zone_upper_seg[1], zone_upper_seg[0]],
        axis=0,
    )
    lower_ci_polygon = segment_strip(zone_lower_seg, root_axis_direction, ci_half_width_px)
    upper_ci_polygon = segment_strip(zone_upper_seg, root_axis_direction, ci_half_width_px)

    mask_layer = canvas.copy()
    cv2.fillPoly(mask_layer, [as_int_points(root_polygon)], color=(170, 220, 40))
    canvas = cv2.addWeighted(mask_layer, 0.22, canvas, 0.78, 0.0)
    cv2.polylines(canvas, [as_int_points(root_polygon)], isClosed=True, color=(170, 220, 40), thickness=2)

    zone_layer = canvas.copy()
    cv2.fillPoly(zone_layer, [as_int_points(zone_polygon)], color=(80, 235, 50))
    cv2.fillPoly(zone_layer, [as_int_points(lower_ci_polygon)], color=(60, 60, 255))
    cv2.fillPoly(zone_layer, [as_int_points(upper_ci_polygon)], color=(60, 60, 255))
    canvas = cv2.addWeighted(zone_layer, 0.28, canvas, 0.72, 0.0)

    axis_start = annulus_mid - root_axis_direction * axis_before_ring_px
    axis_end = stj_mid + root_axis_direction * axis_beyond_stj_px
    cv2.line(canvas, tuple(np.round(aa1).astype(int)), tuple(np.round(aa2).astype(int)), (220, 0, 255), 3)
    cv2.line(
        canvas,
        tuple(np.round(zone_lower_seg[0]).astype(int)),
        tuple(np.round(zone_lower_seg[1]).astype(int)),
        (70, 70, 255),
        2,
    )
    cv2.line(
        canvas,
        tuple(np.round(zone_upper_seg[0]).astype(int)),
        tuple(np.round(zone_upper_seg[1]).astype(int)),
        (70, 70, 255),
        2,
    )
    cv2.arrowedLine(
        canvas,
        tuple(np.round(axis_start).astype(int)),
        tuple(np.round(axis_end).astype(int)),
        (255, 120, 40),
        3,
        tipLength=0.08,
    )
    return canvas


def draw_points(canvas: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    for idx, name in enumerate(POINT_NAMES):
        x, y = np.round(points_xy[idx]).astype(int)
        cv2.circle(canvas, (x, y), 6, (30, 230, 255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 6, (20, 20, 20), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas,
            name,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def render_overlay(
    image_bgr: np.ndarray,
    mask_binary: np.ndarray,
    points_xy: np.ndarray,
) -> np.ndarray:
    overlay = image_bgr.copy()
    color_mask = mask_to_overlay(mask_binary)
    overlay = cv2.addWeighted(color_mask, 0.24, overlay, 0.76, 0.0)
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (40, 220, 170), 2)
    overlay = draw_zone_geometry(overlay, points_xy)
    overlay = draw_points(overlay, points_xy)
    return overlay


def resolve_images(path: Path) -> List[Path]:
    def _resolve_missing_image(candidate: Path) -> Path | None:
        name = candidate.name
        sibling_candidates: List[Path] = []
        if not name.startswith("orig_"):
            sibling_candidates.append(candidate.parent / f"orig_{name}")
        else:
            sibling_candidates.append(candidate.parent / name.removeprefix("orig_"))

        dataset_name = name.removeprefix("orig_")
        patient_token = dataset_name.split("_", 1)[0]
        if patient_token.isdigit():
            patient_dir = f"{int(patient_token):03d}"
            sibling_candidates.append(
                get_data_path("export_project", "segmentation_point", patient_dir, "img", dataset_name)
            )

        for alt in sibling_candidates:
            if alt.is_file():
                print(f"[INFO] Input image not found, using fallback '{alt}'.")
                return alt
        return None

    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTS:
            raise RuntimeError(f"Unsupported image extension: '{path}'.")
        return [path]
    if not path.exists():
        if path.suffix.lower() in IMAGE_EXTS:
            resolved = _resolve_missing_image(path)
            if resolved is not None:
                return [resolved]
        raise RuntimeError(f"Input path does not exist: '{path}'.")
    if not path.is_dir():
        raise RuntimeError(f"Input path is neither file nor directory: '{path}'.")
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file())


@torch.no_grad()
def run_inference(
    model: BoundaryAwareMAnet,
    image_path: Path,
    output_dir: Path,
    img_size: int,
    threshold: float,
    device: torch.device,
    softargmax_beta: float,
) -> Path:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image '{image_path}'.")

    input_tensor, orig_size = prepare_image(image_bgr, img_size)
    input_tensor = input_tensor.to(device=device, dtype=torch.float32)

    out = model(input_tensor, guidance_weight=0.0)
    seg_logits = out["segmentation"]
    heatmaps = out["point_heatmaps"]

    orig_h, orig_w = orig_size
    seg_prob = torch.sigmoid(seg_logits)
    seg_prob = F.interpolate(seg_prob, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    mask_binary = (seg_prob[0, 0].detach().cpu().numpy() >= threshold).astype(np.uint8) * 255
    points_xy = predict_points(heatmaps, orig_size, beta=softargmax_beta)

    rendered = render_overlay(image_bgr, mask_binary, points_xy)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_implant_zone.png"
    cv2.imwrite(str(output_path), rendered)
    return output_path


def load_weights(model: BoundaryAwareMAnet, weights_path: Path, device: torch.device) -> None:
    try:
        checkpoint = load_checkpoint(weights_path, device)
        resolved_weights = weights_path
    except RuntimeError as exc:
        fallback = find_fallback_checkpoint(weights_path)
        if fallback is None:
            raise
        print(f"[WARN] {exc}")
        print(f"[WARN] Falling back to checkpoint '{fallback}'.")
        checkpoint = load_checkpoint(fallback, device)
        resolved_weights = fallback

    state_dict = strip_prefixes(extract_state_dict(checkpoint))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] Loaded weights from '{resolved_weights}'.")
    if missing:
        print(f"[WARN] Missing keys while loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys while loading checkpoint: {len(unexpected)}")


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    if args.weights is None:
        raise FileNotFoundError(
            "Weights not specified and no default checkpoint was found. "
            "Pass --weights /path/to/best.pt or --weights /path/to/checkpoints/*.ckpt."
        )
    weights_path = args.weights.expanduser().resolve()
    input_path = args.input_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: '{config_path}'.")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: '{weights_path}'.")

    cfg = load_yaml(config_path)
    img_size = int(args.img_size or cfg.get("img_size", 896))
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    softargmax_beta = infer_softargmax_beta(cfg)

    images = resolve_images(input_path)
    if not images:
        raise RuntimeError(f"No images found in '{input_path}'.")

    model = build_model(cfg, device)
    load_weights(model, weights_path, device)

    for image_path in images:
        out_path = run_inference(
            model=model,
            image_path=image_path,
            output_dir=output_dir,
            img_size=img_size,
            threshold=args.mask_threshold,
            device=device,
            softargmax_beta=softargmax_beta,
        )
        print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
