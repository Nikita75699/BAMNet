import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
import torchvision.transforms.functional as TF
from skimage.measure import label
import pytorch_lightning as pl
from pathlib import Path

# ==========================================
# CONSTANTS
# ==========================================
IMG_SIZE = 640 # Как указано в config_v4.yaml
SPACING = 0.39  # Pixel spacing (mm/pixel). Если у вас есть данные mm/px (Dicom/NIfTI), пропишите их здесь.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SCORE_THRESHOLD = 0.5  # Берем пиксели со score > 50%
# ==========================================

def get_largest_connected_component(mask):
    """Оставляет только самую большую связную область, убирая мелкий мусор."""
    if not np.any(mask):
        return mask
    labels = label(mask)
    if labels.max() == 0:
        return mask
    largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largest_cc.astype(np.uint8)

def get_boundary(mask):
    """Возвращает только граничные пиксели маски (поверхность)."""
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    import scipy.ndimage as ndi
    eroded = ndi.binary_erosion(mask, structure=np.ones((3,3)))
    return (mask.astype(bool) ^ eroded)

def compute_distances(mask_gt, mask_pred, spacing_mm):
    """Считает расстояния между границами (поверхностями) GT и Pred."""
    bound_gt = get_boundary(mask_gt)
    bound_pred = get_boundary(mask_pred)
    
    if not np.any(bound_gt) and not np.any(bound_pred):
        return np.array([]), np.array([]), bound_pred, bound_gt
    if not np.any(bound_gt) or not np.any(bound_pred):
        # Штраф, если маска полностью пропущена
        return np.array([1000.0]), np.array([1000.0]), bound_pred, bound_gt
        
    dt_gt = distance_transform_edt(~bound_gt, sampling=spacing_mm)
    dt_pred = distance_transform_edt(~bound_pred, sampling=spacing_mm)
    
    dist_pred_to_gt = dt_gt[bound_pred]
    dist_gt_to_pred = dt_pred[bound_gt]
    
    return dist_pred_to_gt, dist_gt_to_pred, bound_pred, bound_gt

def load_manet_model(checkpoint_path):
    import sys
    sys.path.insert(0, str(Path(__file__).parent / "MANet"))
    import train
    modules = train.import_training_modules()
    _, LightningSegmentationModel = train.build_lightning_classes(modules)
    
    model = LightningSegmentationModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.freeze()
    return model

def main():
    checkpoint_path = r"C:\Users\nikit\Downloads\BAMNet\MANet\20260314_192026\checkpoints\best-44-0.0000.ckpt"
    val_images_dir = r"C:\Users\nikit\Downloads\BAMNet\MANet\fold_from_new\val"
    val_masks_dir = r"C:\Users\nikit\Downloads\BAMNet\MANet\fold_from_new\valannot"
    
    if not os.path.exists(val_images_dir):
        print(f"Waiting for data... {val_images_dir} doesn't exist.")
        return

    model = load_manet_model(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    image_files = sorted([f for f in os.listdir(val_images_dir) if f.endswith(".png")])
    
    # Глобальные счетчики для объема ("метрики не для каждого изображения а в общем")
    global_inter = 0
    global_union = 0
    global_pred_sum = 0
    global_gt_sum = 0

    all_dists_pred_to_gt = []
    all_dists_gt_to_pred = []
    
    # Для Surface Dice (Tolerance = 4.0 мм/пикс)
    TOLERANCE = 4.0
    tolerant_pred_in_gt = 0
    tolerant_gt_in_pred = 0
    total_pred_bound = 0
    total_gt_bound = 0

    results_per_image = []

    for img_name in tqdm(image_files, desc="Evaluating"):
        img_path = os.path.join(val_images_dir, img_name)
        mask_path = os.path.join(val_masks_dir, img_name)
        
        if not os.path.exists(mask_path):
            continue
            
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt = (gt > 0).astype(np.uint8)
        
        h, w = gt.shape
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        
        # ОЧЕНЬ ВАЖНО: нормализация ImageNet чтобы избежать шума
        img_tensor = TF.normalize(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(img_tensor)
            if isinstance(logits, dict):
                logits = logits["segmentation"]
            probs = torch.sigmoid(logits)
            # Проверка, что берутся пиксели с probability > 0.5 (score > 50)
            pred = (probs > SCORE_THRESHOLD).cpu().numpy()[0, 0].astype(np.uint8)
        
        # Оставляем только самую большую структуру (аорту)
        pred = get_largest_connected_component(pred)
        
        pred_resized = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # 1. Глобальный Dice/IoU: собираем пересечения
        inter = np.logical_and(pred_resized, gt).sum()
        union = np.logical_or(pred_resized, gt).sum()
        pred_sum = pred_resized.sum()
        gt_sum = gt.sum()
        
        global_inter += inter
        global_union += union
        global_pred_sum += pred_sum
        global_gt_sum += gt_sum

        # 2. Дистанции между ПОВЕРХНОСТЯМИ (границы)
        dist_pred_to_gt, dist_gt_to_pred, bound_pred, bound_gt = compute_distances(gt, pred_resized, SPACING)
        
        all_dists_pred_to_gt.extend(dist_pred_to_gt)
        all_dists_gt_to_pred.extend(dist_gt_to_pred)
        
        tolerant_pred_in_gt += np.sum(dist_pred_to_gt <= TOLERANCE)
        tolerant_gt_in_pred += np.sum(dist_gt_to_pred <= TOLERANCE)
        total_pred_bound += np.sum(bound_pred)
        total_gt_bound += np.sum(bound_gt)

        # Метрики считаются чисто для таблицы/лога по каждому снимку
        img_dice = (2. * inter) / (pred_sum + gt_sum + 1e-7)
        img_iou = inter / (union + 1e-7)
        img_dists = np.concatenate([dist_pred_to_gt, dist_gt_to_pred]) if len(dist_pred_to_gt) > 0 else np.array([])
        img_hd95 = np.percentile(img_dists, 95) if len(img_dists) > 0 else 0.0
        img_assd = np.mean(img_dists) if len(img_dists) > 0 else 0.0
        
        results_per_image.append({
            "image": img_name,
            "dice": img_dice,
            "iou": img_iou,
            "hd95": img_hd95,
            "assd": img_assd
        })

    # ==========================================
    # ВЫЧИСЛЕНИЕ ОБЩЕЙ МЕТРИКИ (GLOBAL)
    # ==========================================
    global_dice = 2.0 * global_inter / (global_pred_sum + global_gt_sum + 1e-7)
    global_iou = global_inter / (global_union + 1e-7)
    
    all_dists = np.concatenate([all_dists_pred_to_gt, all_dists_gt_to_pred]) if all_dists_pred_to_gt else np.array([])
    if len(all_dists) > 0:
        global_hd95 = np.percentile(all_dists, 95)
        global_assd = np.mean(all_dists)
    else:
        global_hd95 = 0.0
        global_assd = 0.0

    global_surface_dice = (tolerant_pred_in_gt + tolerant_gt_in_pred) / (total_pred_bound + total_gt_bound + 1e-7)

    print("\n" + "="*50)
    print("GLOBAL DATASET METRICS (В общем, а не усреднение картинок):")
    print("="*50)
    print(f"Global Dice:  {global_dice:.4f}")
    print(f"Global IoU:   {global_iou:.4f}")
    print(f"Global HD95:  {global_hd95:.4f} (в пикселях - умножьте на mm/px)")
    print(f"Global ASSD:  {global_assd:.4f} (в пикселях - умножьте на mm/px)")
    print(f"Global SurfD: {global_surface_dice:.4f} (Толерантность: ±{TOLERANCE} px)")
    print("="*50)
    
    df = pd.DataFrame(results_per_image)
    df.to_csv("manet_evaluation_results.csv", index=False)
    print("\nPer-image results saved to manet_evaluation_results.csv")

if __name__ == "__main__":
    main()
