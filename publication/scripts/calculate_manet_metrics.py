import argparse
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, label
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bamnet_paths import get_data_path


# ==========================================
# CONSTANTS
# ==========================================
IMG_SIZE = 640  # Matches the current root training config.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SCORE_THRESHOLD = 0.5  # Берем пиксели со score > 50%
SURFACE_DICE_TOLERANCE_MM = 4.0
# ==========================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MANet segmentation quality with per-image pixel spacing from Supervisely img_info."
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--val-images-dir", type=Path, required=True)
    parser.add_argument("--val-masks-dir", type=Path, required=True)
    parser.add_argument(
        "--supervisely-root",
        type=Path,
        default=get_data_path("export_project", "segmentation_point"),
        help="Path to raw Supervisely export with per-patient img_info/*.json metadata.",
    )
    parser.add_argument(
        "--manet-root",
        type=Path,
        default=ROOT_DIR / "MANet",
        help="Path to the MANet training repository used to load the checkpoint.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("manet_evaluation_results.csv"),
    )
    return parser.parse_args()


def get_largest_connected_component(mask):
    """Оставляет только самую большую связную область, убирая мелкий мусор."""
    if not np.any(mask):
        return mask
    labels, num_features = label(mask)
    if num_features == 0:
        return mask
    largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largest_cc.astype(np.uint8)


def get_boundary(mask):
    """Возвращает только граничные пиксели маски (поверхность)."""
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    import scipy.ndimage as ndi

    eroded = ndi.binary_erosion(mask, structure=np.ones((3, 3)))
    return mask.astype(bool) ^ eroded


def compute_distances(mask_gt, mask_pred, spacing_mm):
    """Считает расстояния между границами (поверхностями) GT и Pred в миллиметрах."""
    bound_gt = get_boundary(mask_gt)
    bound_pred = get_boundary(mask_pred)

    if not np.any(bound_gt) and not np.any(bound_pred):
        return np.array([]), np.array([]), bound_pred, bound_gt
    if not np.any(bound_gt) or not np.any(bound_pred):
        # Штраф, если маска полностью пропущена.
        return np.array([1000.0]), np.array([1000.0]), bound_pred, bound_gt

    dt_gt = distance_transform_edt(~bound_gt, sampling=spacing_mm)
    dt_pred = distance_transform_edt(~bound_pred, sampling=spacing_mm)

    dist_pred_to_gt = dt_gt[bound_pred]
    dist_gt_to_pred = dt_pred[bound_gt]

    return dist_pred_to_gt, dist_gt_to_pred, bound_pred, bound_gt


def load_manet_model(checkpoint_path: Path, manet_root: Path):
    sys.path.insert(0, str(manet_root.resolve()))
    import train

    modules = train.import_training_modules()
    _, LightningSegmentationModel = train.build_lightning_classes(modules)

    model = LightningSegmentationModel.load_from_checkpoint(str(checkpoint_path))
    model.eval()
    model.freeze()
    return model


def patient_dir_from_image_name(image_name: str) -> str:
    patient_token = image_name.split("_", 1)[0]
    if not patient_token.isdigit():
        raise ValueError(f"Cannot derive patient id from image name: {image_name}")
    return f"{int(patient_token):03d}"


@lru_cache(maxsize=None)
def load_pixel_spacing_row_mm(supervisely_root_str: str, image_name: str) -> float:
    supervisely_root = Path(supervisely_root_str)
    patient_dir = patient_dir_from_image_name(image_name)
    img_info_path = supervisely_root / patient_dir / "img_info" / f"{image_name}.json"

    if not img_info_path.is_file():
        raise FileNotFoundError(
            f"img_info metadata not found for {image_name}: {img_info_path}"
        )

    payload = json.loads(img_info_path.read_text(encoding="utf-8"))
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        raise KeyError(f"Missing 'meta' block in {img_info_path}")

    spacing = meta.get("pixel_spacing_row_mm")
    if spacing is None:
        raise KeyError(f"Missing meta.pixel_spacing_row_mm in {img_info_path}")

    return float(spacing)


def main():
    import cv2
    import pandas as pd
    import torch
    import torchvision.transforms.functional as TF

    args = parse_args()
    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    val_images_dir = args.val_images_dir.expanduser().resolve()
    val_masks_dir = args.val_masks_dir.expanduser().resolve()
    supervisely_root = args.supervisely_root.expanduser().resolve()
    manet_root = args.manet_root.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()

    if not val_images_dir.exists():
        raise FileNotFoundError(f"val images dir does not exist: {val_images_dir}")
    if not val_masks_dir.exists():
        raise FileNotFoundError(f"val masks dir does not exist: {val_masks_dir}")
    if not supervisely_root.exists():
        raise FileNotFoundError(f"Supervisely root does not exist: {supervisely_root}")
    if not manet_root.exists():
        raise FileNotFoundError(f"MANet root does not exist: {manet_root}")

    model = load_manet_model(checkpoint_path, manet_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    image_files = sorted([f for f in os.listdir(val_images_dir) if f.endswith(".png")])

    # Глобальные счетчики для объема ("метрики не для каждого изображения, а в общем")
    global_inter = 0
    global_union = 0
    global_pred_sum = 0
    global_gt_sum = 0

    all_dists_pred_to_gt = []
    all_dists_gt_to_pred = []

    tolerant_pred_in_gt = 0
    tolerant_gt_in_pred = 0
    total_pred_bound = 0
    total_gt_bound = 0

    results_per_image = []

    for img_name in tqdm(image_files, desc="Evaluating"):
        img_path = val_images_dir / img_name
        mask_path = val_masks_dir / img_name

        if not mask_path.exists():
            continue

        spacing_mm = load_pixel_spacing_row_mm(str(supervisely_root), img_name)

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gt = (gt > 0).astype(np.uint8)

        h, w = gt.shape
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = TF.normalize(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            if isinstance(logits, dict):
                logits = logits["segmentation"]
            probs = torch.sigmoid(logits)
            pred = (probs > SCORE_THRESHOLD).cpu().numpy()[0, 0].astype(np.uint8)

        pred = get_largest_connected_component(pred)
        pred_resized = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

        inter = np.logical_and(pred_resized, gt).sum()
        union = np.logical_or(pred_resized, gt).sum()
        pred_sum = pred_resized.sum()
        gt_sum = gt.sum()

        global_inter += inter
        global_union += union
        global_pred_sum += pred_sum
        global_gt_sum += gt_sum

        dist_pred_to_gt, dist_gt_to_pred, bound_pred, bound_gt = compute_distances(
            gt,
            pred_resized,
            spacing_mm,
        )

        all_dists_pred_to_gt.extend(dist_pred_to_gt)
        all_dists_gt_to_pred.extend(dist_gt_to_pred)

        tolerant_pred_in_gt += np.sum(dist_pred_to_gt <= SURFACE_DICE_TOLERANCE_MM)
        tolerant_gt_in_pred += np.sum(dist_gt_to_pred <= SURFACE_DICE_TOLERANCE_MM)
        total_pred_bound += np.sum(bound_pred)
        total_gt_bound += np.sum(bound_gt)

        img_dice = (2.0 * inter) / (pred_sum + gt_sum + 1e-7)
        img_iou = inter / (union + 1e-7)
        img_dists = (
            np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
            if len(dist_pred_to_gt) > 0
            else np.array([])
        )
        img_hd95 = np.percentile(img_dists, 95) if len(img_dists) > 0 else 0.0
        img_assd = np.mean(img_dists) if len(img_dists) > 0 else 0.0

        results_per_image.append(
            {
                "image": img_name,
                "pixel_spacing_row_mm": spacing_mm,
                "dice": img_dice,
                "iou": img_iou,
                "hd95_mm": img_hd95,
                "assd_mm": img_assd,
            }
        )

    global_dice = 2.0 * global_inter / (global_pred_sum + global_gt_sum + 1e-7)
    global_iou = global_inter / (global_union + 1e-7)

    all_dists = (
        np.concatenate([all_dists_pred_to_gt, all_dists_gt_to_pred])
        if all_dists_pred_to_gt
        else np.array([])
    )
    if len(all_dists) > 0:
        global_hd95 = np.percentile(all_dists, 95)
        global_assd = np.mean(all_dists)
    else:
        global_hd95 = 0.0
        global_assd = 0.0

    global_surface_dice = (tolerant_pred_in_gt + tolerant_gt_in_pred) / (
        total_pred_bound + total_gt_bound + 1e-7
    )

    print("\n" + "=" * 50)
    print("GLOBAL DATASET METRICS (в общем, а не усреднение картинок):")
    print("=" * 50)
    print(f"Global Dice:  {global_dice:.4f}")
    print(f"Global IoU:   {global_iou:.4f}")
    print(f"Global HD95:  {global_hd95:.4f} mm")
    print(f"Global ASSD:  {global_assd:.4f} mm")
    print(
        f"Global SurfD: {global_surface_dice:.4f} "
        f"(Tolerance: ±{SURFACE_DICE_TOLERANCE_MM:.1f} mm)"
    )
    print("=" * 50)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results_per_image)
    df.to_csv(output_csv, index=False)
    print(f"\nPer-image results saved to {output_csv}")


if __name__ == "__main__":
    main()
