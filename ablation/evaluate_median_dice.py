import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Add root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model_backbone_coords import LitBoundaryAwareSystem, dice_coef
from data import CustomDataModule

def find_best_pt(variant_dir):
    # Try direct
    p = variant_dir / "best.pt"
    if p.exists(): return p
    # Try recursive
    for p in variant_dir.rglob("best.pt"):
        return p
    return None

def evaluate_variant(variant_dir, data_path, device="cpu"):
    variant_dir = Path(variant_dir)
    ckpt_path = find_best_pt(variant_dir)
    
    if ckpt_path is None:
        print(f"[WARN] No best.pt found in {variant_dir}")
        return None
    
    # Load model
    # Note: Lazy layers require weights_only=False in newer PyTorch
    model = LitBoundaryAwareSystem.load_from_checkpoint(
        ckpt_path, 
        map_location=device,
        weights_only=False
    )
    model.to(device)
    model.eval()
    
    # Setup data: direct initialization to avoid needing 'train' folder
    from torch.utils.data import DataLoader
    from data import CustomDataset
    
    val_dataset = CustomDataset(
        data_path=os.path.join(data_path, "val"),
        img_size=512,
        augment=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0 if device == "cpu" else 4,
        pin_memory=True if device != "cpu" else False
    )
    
    dice_scores = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Evaluating {variant_dir.name}", leave=False):
            x, targets = batch
            x = x.to(device)
            mask_tgt = targets["mask"].to(device)
            
            out = model(x)
            seg_logits = out["segmentation"]
            
            # Align mask to logits size (e.g. 256 if input is 512)
            h, w = seg_logits.shape[-2:]
            mask_tgt_down = torch.nn.functional.interpolate(
                mask_tgt, size=(h, w), mode="nearest"
            )
            
            # Binarize targets
            mask_tgt_bin = (mask_tgt_down > 0.5).float()
            
            # Predict
            pred_probs = torch.sigmoid(seg_logits)
            pred_bin = (pred_probs > 0.5).float()
            
            # Calculate per-sample dice
            inter = (pred_bin * mask_tgt_bin).sum(dim=(2, 3))
            denom = (pred_bin.sum(dim=(2, 3)) + mask_tgt_bin.sum(dim=(2, 3))).clamp_min(1e-6)
            dice_batch = (2 * inter / denom).squeeze(1) # (B,)
            dice_scores.extend(dice_batch.cpu().numpy().tolist())
            
    return np.median(dice_scores)

def main():
    ablation_root = ROOT_DIR / "ablation" / "runs_ablation"
    data_root = str(ROOT_DIR) # Root contains 'val'
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    variants = [
        "ablation_v4_full_model",
        "ablation_v4_no_position_attention",
        "ablation_v4_no_coordinate_attention",
        "ablation_v4_no_boundary_loss",
        "ablation_v4_no_fusion",
        "ablation_v4_no_boundary_guidance",
        "ablation_v4_beta_fixed_8",
        "ablation_v4_beta_schedule_4_8",
    ]
    
    results = {}
    
    for v in variants:
        v_dir = ablation_root / v
        if v_dir.exists():
            median_dice = evaluate_variant(v_dir, data_root, device=device)
            if median_dice is not None:
                results[v] = median_dice
                print(f"Variant: {v}, Median Dice: {median_dice:.4f}")
            else:
                print(f"Variant: {v}, Median Dice: N/A")
        else:
            print(f"[ERR] Variant directory not found: {v_dir}")
    
    # Print summary for copying
    print("\nSummary results (Median Dice):")
    for v in variants:
        val = results.get(v)
        res_str = f"{val:.4f}" if val is not None else "N/A"
        print(f"{v}: {res_str}")

if __name__ == "__main__":
    main()
