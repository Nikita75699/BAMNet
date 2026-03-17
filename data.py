import os
import json
import inspect
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import albumentations as A

cv2.setNumThreads(0)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_AUG_KP_WARN_COUNT = 0


def _albumentations_major_version() -> int:
    version = getattr(A, "__version__", "1")
    try:
        return int(str(version).split(".", 1)[0])
    except Exception:
        return 1


def _supports_kwarg(callable_obj, kwarg_name: str) -> bool:
    try:
        sig = inspect.signature(callable_obj)
    except Exception:
        return False
    return kwarg_name in sig.parameters


def _normalize_aug_keypoints(
    keypoints_out,
    num_points: int,
    fallback_pts_xy: np.ndarray,
) -> np.ndarray:
    """Приводит keypoints из Albumentations к форме (num_points, 2).

    В некоторых версиях/трансформах Albumentations keypoints могут вернуться:
    - как (N, 2)
    - как (N, K), где K > 2 (доп. служебные поля)
    - как плоский вектор (num_points * K), где K > 2
    """
    kps_arr = np.asarray(keypoints_out, dtype=np.float32)

    if num_points <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    if kps_arr.size == 0:
        return fallback_pts_xy.astype(np.float32, copy=True)

    if kps_arr.ndim == 1:
        # Предпочитаем восстановление по числу точек, т.к. некоторые версии возвращают плоский (P * K).
        if (kps_arr.size % num_points) == 0 and (kps_arr.size // num_points) >= 2:
            kps_arr = kps_arr.reshape(num_points, -1)
        elif (kps_arr.size % 2) == 0:
            kps_arr = kps_arr.reshape(-1, 2)
        else:
            raise ValueError(f"Unexpected flat keypoints size: {kps_arr.size}")

    if kps_arr.ndim != 2:
        raise ValueError(f"Unexpected keypoints ndim: {kps_arr.ndim}")
    if kps_arr.shape[1] < 2:
        raise ValueError(f"Unexpected keypoints shape: {kps_arr.shape}")

    # Некоторые версии/трансформы возвращают "расширенный" формат как (P*K, M),
    # где P — число точек, K — число служебных групп. Восстанавливаем по общему size.
    if kps_arr.shape[0] != num_points and (kps_arr.size % num_points) == 0 and (kps_arr.size // num_points) >= 2:
        kps_arr = kps_arr.reshape(num_points, -1)

    kps_xy = kps_arr[:, :2]

    if kps_xy.shape[0] == num_points:
        return kps_xy.astype(np.float32, copy=False)

    # Fallback: сохраняем исходный порядок/число точек, если трансформ вернул неожиданный count.
    global _AUG_KP_WARN_COUNT
    _AUG_KP_WARN_COUNT += 1
    if _AUG_KP_WARN_COUNT <= 5:
        print(
            f"[WARN] Albumentations returned unexpected keypoint count: "
            f"{kps_xy.shape[0]} (expected {num_points}). Fallback to original points."
        )
        if _AUG_KP_WARN_COUNT == 5:
            print("[WARN] Further Albumentations keypoint warnings will be suppressed.")
    return fallback_pts_xy.astype(np.float32, copy=True)


def _build_train_aug(img_size: int) -> A.Compose:
    """Единый Albumentations-пайплайн: геометрия + фотометрия + keypoints."""
    coarse_h = max(8, int(0.10 * img_size))
    coarse_w = max(8, int(0.10 * img_size))

    # Надёжнее версии: определяем поддержку kwargs по сигнатуре.
    elastic_kwargs = {"alpha": 1, "sigma": 50, "p": 0.3}
    if _supports_kwarg(A.ElasticTransform.__init__, "alpha_affine"):
        elastic_kwargs["alpha_affine"] = 50
    elastic = A.ElasticTransform(**elastic_kwargs)

    if _supports_kwarg(A.GaussNoise.__init__, "var_limit"):
        gauss_noise = A.GaussNoise(var_limit=(10.0, 50.0), p=0.2)
    else:
        gauss_noise = A.GaussNoise(std_range=(0.02, 0.08), mean_range=(0.0, 0.0), p=0.2)

    if _supports_kwarg(A.CoarseDropout.__init__, "max_holes"):
        coarse_dropout = A.CoarseDropout(
            max_holes=4,
            max_height=coarse_h,
            max_width=coarse_w,
            min_holes=1,
            fill_value=0,
            p=0.2,
        )
    else:
        coarse_dropout = A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(coarse_h, coarse_h),
            hole_width_range=(coarse_w, coarse_w),
            fill=0,
            p=0.2,
        )

    return A.Compose(
        [
            # ----- геометрия -------------------------------------------------
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.90, 1.10),
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                rotate=(-15, 15),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.7,
            ),
            elastic,  # имитация изгибов сосудов

            # ----- фотометрия -----------------------------------------------
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            gauss_noise,  # шум матрицы
            coarse_dropout,
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

def apply_old_geom_aug(image, mask, pts_xy, vis_flags):
    """
    image: uint8 RGB (H,W,3)
    mask:  uint8 (H,W)
    pts_xy: (P,2) float32 in pixels
    vis_flags: (P,) float32 {0,1}
    """
    H, W = image.shape[:2]

    do_hflip = np.random.rand() < 0.5
    do_vflip = False
    angle = np.random.uniform(-15.0, 15.0)
    scale = 1.0 + np.random.uniform(-0.10, 0.10)
    tx = np.random.uniform(-0.05, 0.05) * W
    ty = np.random.uniform(-0.05, 0.05) * H

    cx, cy = (W - 1) * 0.5, (H - 1) * 0.5

    M_rs = cv2.getRotationMatrix2D((cx, cy), angle, scale).astype(np.float32)

    sx = -1.0 if do_hflip else 1.0
    sy = -1.0 if do_vflip else 1.0
    S = np.array([[sx, 0, cx * (1 - sx)],
                  [0, sy, cy * (1 - sy)]], dtype=np.float32)

    R3 = np.vstack([M_rs, [0., 0., 1.]]).astype(np.float32)
    S3 = np.vstack([S,    [0., 0., 1.]]).astype(np.float32)

    # сначала flip, потом rotate/scale
    M = (R3 @ S3)[:2, :]
    M[:, 2] += np.array([tx, ty], dtype=np.float32)

    image_aug = cv2.warpAffine(
        image, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    mask_aug = cv2.warpAffine(
        mask, M, (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    if pts_xy.shape[0] > 0:
        pts_h = np.concatenate([pts_xy, np.ones((pts_xy.shape[0], 1), dtype=np.float32)], axis=1)  # (P,3)
        pts_new = (pts_h @ M.T).astype(np.float32)  # (P,2)

        # Если точка улетела за кадр — делаем невидимой (важно!)
        inb = (
            (pts_new[:, 0] >= 0) & (pts_new[:, 0] <= W - 1) &
            (pts_new[:, 1] >= 0) & (pts_new[:, 1] <= H - 1)
        ).astype(np.float32)
        vis_flags = vis_flags * inb

        # клип уже после inb
        pts_new[:, 0] = np.clip(pts_new[:, 0], 0.0, W - 1)
        pts_new[:, 1] = np.clip(pts_new[:, 1], 0.0, H - 1)
    else:
        pts_new = pts_xy

    return image_aug, mask_aug, pts_new, vis_flags

class CustomDataset(Dataset):
    def __init__(self, data_path: str, img_size: int = 512,
                 point_names=None, augment: bool = True, debug: bool = False):
        self.img_size = int(img_size)
        self.data_path = data_path
        self.point_names = point_names or ["AA1", "AA2"]
        self.num_points = len(self.point_names)
        self.augment = augment
        self.debug = debug

        self.images_dir = os.path.join(data_path, "images")
        self.masks_dir = os.path.join(data_path, "masks")
        self.points_dir = os.path.join(data_path, "points")
        
        self.image_files = self._collect_image_files()
        self.length = len(self.image_files)
        
        if self.debug:
            print(f"[INFO] found {self.length} images in {self.images_dir}")
        self.aug = None  # геометрию делаем вручную как раньше

        self.photo_aug = A.Compose([
            A.RandomBrightnessContrast(p=0.3),
            A.ColorJitter(p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.CoarseDropout(
                max_holes=2,
                max_height=int(0.12 * self.img_size),
                max_width=int(0.12 * self.img_size),
                min_holes=1,
                fill_value=0,
                p=0.15
            ),
        ]) if self.augment else None

    def _collect_image_files(self):
        image_files = []
        split_name = Path(self.data_path).name.lower()
        is_train_split = split_name == "train"
        is_val_split = split_name == "val"

        for f in os.listdir(self.images_dir):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                base = os.path.splitext(f)[0]
                if os.path.isdir(self.points_dir):
                    has_pts = os.path.exists(os.path.join(self.points_dir, base + ".json"))
                else:
                    has_pts = False
                # для train берём все, для val — только с точками
                if (is_val_split and has_pts) or is_train_split:
                    image_files.append(f)
        return sorted(image_files)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        base_name = os.path.splitext(image_file)[0]

        # --- Load image (grayscale or RGB) ---
        image_path = os.path.join(self.images_dir, image_file)
        raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # uint8, (H,W,3) или (H,W)
        if raw is None:
            raise FileNotFoundError(f"failed to read image: {image_path}")

        # к RGB uint8 (аугментации Albumentations ниже ожидают uint8 для CLAHE/шумов)
        if raw.ndim == 2:
            if raw.dtype != np.uint8:
                raw = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            image = np.repeat(raw[..., None], 3, axis=-1)
        else:
            if raw.dtype != np.uint8:
                raw = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            image = raw[..., ::-1].copy()  # BGR→RGB, uint8

        orig_h, orig_w = image.shape[:2]

        # --- Load mask ---
        mask_path = os.path.join(self.masks_dir, base_name + ".png")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            else:
                if mask.max() > 1:
                    mask = (mask > 0).astype(np.uint8) * 255
        else:
            mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        # --- Load points (robust, prefer absolute x/y) ---
        points_path = os.path.join(self.points_dir, base_name + ".json")
        points_vec = np.zeros(self.num_points * 3, dtype=np.float32)

        if os.path.exists(points_path):
            try:
                with open(points_path, "r") as fh:
                    points_data = json.load(fh)

                container = points_data.get("points", points_data)
                js_w = js_h = None
                img_size = points_data.get("image_size") or points_data.get("size")
                if isinstance(img_size, dict):
                    js_w = img_size.get("w") or img_size.get("width")
                    js_h = img_size.get("h") or img_size.get("height")

                for j, name in enumerate(self.point_names):
                    entry = container.get(name)
                    if not entry:
                        continue

                    vis = entry.get("visible", None)
                    x = entry.get("x", None); y = entry.get("y", None)
                    xn = entry.get("x_norm", None); yn = entry.get("y_norm", None)

                    if x is not None and y is not None:
                        denom_w = float(js_w) if js_w else max(1.0, float(orig_w))
                        denom_h = float(js_h) if js_h else max(1.0, float(orig_h))
                        x_norm = float(x) / denom_w
                        y_norm = float(y) / denom_h
                    elif xn is not None and yn is not None:
                        x_norm = float(xn); y_norm = float(yn)
                    else:
                        x_norm = y_norm = 0.0

                    if vis is None:
                        visible = 1.0 if (x_norm > 0.0 or y_norm > 0.0) else 0.0
                    else:
                        visible = 1.0 if float(vis) > 0.5 else 0.0

                    points_vec[3*j]   = np.clip(x_norm, 0.0, 1.0)
                    points_vec[3*j+1] = np.clip(y_norm, 0.0, 1.0)
                    points_vec[3*j+2] = visible
            except Exception as e:
                print(f"[WARN] Failed to load points from {points_path}: {e}")

        if points_vec[2::3].sum() == 0:
            points_vec[2::3] = 0.0

        # --- Resize to target size ---
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask,  (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        H, W = self.img_size, self.img_size

        # --- Подготовим точки (нормализованные -> пиксели) ---
        pts_xy = np.zeros((self.num_points, 2), dtype=np.float32)
        vis_flags = np.zeros((self.num_points,), dtype=np.float32)
        for j in range(self.num_points):
            xn = float(points_vec[3*j + 0])
            yn = float(points_vec[3*j + 1])
            vv = float(points_vec[3*j + 2])
            x_px = np.clip(xn, 0.0, 1.0) * (W - 1)
            y_px = np.clip(yn, 0.0, 1.0) * (H - 1)
            pts_xy[j, 0] = x_px
            pts_xy[j, 1] = y_px
            vis_flags[j] = vv

        # --- Единый Albumentations-пайплайн (геометрия + фотометрия) ---
        if self.augment:
            # --- старая геометрия через cv2.warpAffine ---
            image, mask, pts_xy, vis_flags = apply_old_geom_aug(image, mask, pts_xy, vis_flags)
            mask = (mask > 127).astype(np.uint8) * 255 
            # --- фотометрия отдельно (не меняет точки) ---
            if self.photo_aug is not None:
                image = self.photo_aug(image=image)["image"]

        # --- Вернём точки в нормализованный вид [0,1] ---
        for j in range(self.num_points):
            x_px, y_px = float(pts_xy[j, 0]), float(pts_xy[j, 1])
            points_vec[3*j + 0] = x_px / (W - 1) if W > 1 else 0.0
            points_vec[3*j + 1] = y_px / (H - 1) if H > 1 else 0.0
            points_vec[3*j + 2] = vis_flags[j]

        # --- Normalize (ImageNet) ---
        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = torch.from_numpy(np.transpose(image.astype(np.float32), (2, 0, 1))).float()

        # --- Tensors to return ---
        mask_t = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        points = torch.tensor(points_vec, dtype=torch.float32)

        if self.debug and idx < 5:
            vis_n = int((points_vec[2::3] > 0.5).sum())
            print(f"[DEBUG] idx={idx}, file={base_name}, mask_sum={mask_t.sum().item():.1f}, visible_pts={vis_n}/{self.num_points}")

        return image, {"mask": mask_t, "points": points}



    @property
    def num_images(self) -> int:
        return int(self.length)

    @property
    def num_nonempty_masks(self) -> int:
        # Simple implementation - count masks that are not all zeros
        count = 0
        for image_file in self.image_files:
            base_name = os.path.splitext(image_file)[0]
            mask_path = os.path.join(self.masks_dir, base_name + ".png")
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None and mask.max() > 0:
                    count += 1
        return count

import pytorch_lightning as pl

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int = 32, img_size: int = 512,
                 num_workers: int = 4, augment: bool = True,
                 point_names=None, debug: bool = False):
        super().__init__()
        self.data_path = data_path
        self.batch_size = int(batch_size)
        self.img_size = int(img_size)
        self.num_workers = int(num_workers)
        self.augment = augment
        self.point_names = point_names
        self.debug = debug

    def inspect_n_samples(self, n: int = 3, split: str = "train", outdir: str = "debug_inspect"):
        import os, numpy as np, cv2, torch
        os.makedirs(outdir, exist_ok=True)
        ds = self.train_dataset if split == "train" else self.val_dataset
        print(f"[INSPECT] split={split}  n={n}")
        for i in range(min(n, len(ds))):
            img_t, tgt = ds[i]
            m = tgt["mask"]
            # тензор → денорм в [0,1] для сохранения
            img = img_t.clone().float()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            img01 = (img * std + mean).clamp(0,1)
            img_np = (img01.permute(1,2,0).numpy() * 255).astype(np.uint8)
            mask_np = (m.squeeze(0).numpy() * 255).astype(np.uint8)

            # печать статистик
            print(f"  #{i}: img shape={img_np.shape}, dtype={img_np.dtype}, "
                f"min/max={img_np.min()}/{img_np.max()}, mean={img_np.mean():.1f}")
            print(f"      mask unique: {np.unique(mask_np)[:5]} ... nz={int((mask_np>0).sum())}")

            # сохраняем превью: image, mask, overlay
            cv2.imwrite(os.path.join(outdir, f"{split}_{i:02d}_img.png"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(outdir, f"{split}_{i:02d}_mask.png"), mask_np)
            # overlay
            overlay = cv2.addWeighted(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), 1.0,
                                    cv2.applyColorMap(mask_np, cv2.COLORMAP_JET), 0.35, 0.0)
            cv2.imwrite(os.path.join(outdir, f"{split}_{i:02d}_overlay.png"), overlay)

    def setup(self, stage=None):
        self.train_dataset = CustomDataset(
            os.path.join(self.data_path, "train"),
            self.img_size, point_names=self.point_names, augment=self.augment, debug=self.debug
        )
        self.val_dataset = CustomDataset(
            os.path.join(self.data_path, "val"),  # Changed from "test" to "val"
            self.img_size, point_names=self.point_names, augment=False, debug=self.debug
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, pin_memory=True, 
                          persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=True, 
                          persistent_workers=self.num_workers > 0)

    def print_stats(self):
        t = self.train_dataset
        v = self.val_dataset
        def fmt(ds, split):
            n_all = ds.num_images
            n_mask = ds.num_nonempty_masks
            pct = (100.0 * n_mask / max(1, n_all))
            print(f"[STATS] {split}: images={n_all}, with_mask={n_mask} ({pct:.1f}%)")
        fmt(t, "train")
        fmt(v, "val")
