#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BAMNet v4 for ablation experiments.")
    parser.add_argument("--config", type=str, default=str(SCRIPT_DIR / "config_v4.yaml"))
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--w_bce", type=float, default=None, help="Override BCE loss weight.")
    parser.add_argument("--w_dice", type=float, default=None, help="Override Dice loss weight.")
    parser.add_argument("--w_pts", type=float, default=None, help="Override points loss weight.")
    parser.add_argument("--w_bnd", type=float, default=None, help="Override boundary loss weight.")
    parser.add_argument("--exp_name", type=str, default=None, help="Override experiment name.")
    args = parser.parse_args()

    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

    from data import CustomDataModule
    from model_backbone_coords_v4 import build_system

    cfg = load_config(args.config)

    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.w_bce is not None:
        cfg["loss"]["w_bce"] = args.w_bce
    if args.w_dice is not None:
        cfg["loss"]["w_dice"] = args.w_dice
    if args.w_pts is not None:
        cfg["loss"]["w_pts"] = args.w_pts
    if args.w_bnd is not None:
        cfg["loss"]["w_bnd"] = args.w_bnd
    if args.exp_name is not None:
        cfg["logging"]["experiment_name"] = args.exp_name

    pl.seed_everything(cfg.get("seed", 42), workers=True)

    dm = CustomDataModule(
        data_path=cfg["data_path"],
        batch_size=cfg["batch_size"],
        img_size=cfg["img_size"],
        num_workers=cfg.get("num_workers", 4),
        augment=cfg.get("augment", True),
        point_names=cfg.get("point_names", None),
    )
    lit = build_system(cfg)

    trainer_cfg = cfg.get("trainer", {})
    epochs = int(trainer_cfg.get("epochs", 100))
    devices = int(trainer_cfg.get("devices", 1))
    precision_cfg = trainer_cfg.get("precision", "auto")
    precision = "bf16-mixed" if (precision_cfg == "auto" and torch.cuda.is_available()) else precision_cfg

    log_cfg = cfg.get("logging", {})
    save_dir = str(Path(log_cfg.get("save_dir", "runs")).expanduser())
    exp_name = log_cfg.get("experiment_name", "default")
    arch = cfg.get("architecture", "unknown")
    run_version = log_cfg.get("version") or datetime.now().strftime("%Y%m%d_%H%M%S")

    tb_logger = TensorBoardLogger(save_dir=save_dir, name=f"{exp_name}/{arch}", version=run_version)
    csv_logger = CSVLogger(save_dir=save_dir, name=f"{exp_name}/{arch}", version=run_version)

    ckpt_best_balance = ModelCheckpoint(
        filename="best-balance-{epoch:03d}-{val_balance_score:.4f}",
        monitor="val_balance_score",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        default_root_dir=save_dir,
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices,
        precision=precision,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        logger=[tb_logger, csv_logger],
        callbacks=[ckpt_best_balance],
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    setattr(lit, "log_dir_override", tb_logger.log_dir)

    dm.setup("fit")
    dm.print_stats()

    trainer.fit(lit, dm)

    best_path = ckpt_best_balance.best_model_path
    run_dir = Path(tb_logger.log_dir)
    if best_path:
        src = Path(best_path)
        dst = run_dir / "best.pt"
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())
        except Exception:
            shutil.copy2(src, dst)
        print(f"[INFO] Best checkpoint: {src}")
        print(f"[INFO] best.pt placed at: {dst}")
    else:
        print("[WARN] No best_model_path - nothing to link/copy.")


if __name__ == "__main__":
    main()
