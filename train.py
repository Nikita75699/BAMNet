import argparse
import yaml
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from data import CustomDataModule
from pathlib import Path
import shutil
from model_backbone_unetpp_coords import build_system
from datetime import datetime

# Видимость GPU задаётся снаружи через окружение или настройки Lightning.

def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')

    parser.add_argument('--lr', type=float, default=None, help="Override learning rate.")
    parser.add_argument('--w_bce', type=float, default=None, help="Override BCE loss weight.")
    parser.add_argument('--w_dice', type=float, default=None, help="Override Dice loss weight.")
    parser.add_argument('--w_pts', type=float, default=None, help="Override points loss weight.")
    parser.add_argument('--w_bnd', type=float, default=None, help="Override boundary loss weight.")
    parser.add_argument('--exp_name', type=str, default=None, help="Override experiment name.")

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.lr is not None:
        cfg['lr'] = args.lr
    if args.w_bce is not None:
        cfg['loss']['w_bce'] = args.w_bce
    if args.w_dice is not None:
        cfg['loss']['w_dice'] = args.w_dice
    if args.w_pts is not None:
        cfg['loss']['w_pts'] = args.w_pts
    if args.w_bnd is not None:
        cfg['loss']['w_bnd'] = args.w_bnd
    if args.exp_name is not None:
        cfg['logging']['experiment_name'] = args.exp_name

    pl.seed_everything(cfg.get('seed', 42), workers=True)

    # ---------------- Data ----------------
    dm = CustomDataModule(
        data_path=cfg['data_path'],
        batch_size=cfg['batch_size'],
        img_size=cfg['img_size'],
        num_workers=cfg.get('num_workers', 4),
        augment=cfg.get('augment', True),
        point_names=cfg.get('point_names', None)
    )

    # ---------------- Model ----------------
    # lit = build_model_and_module(cfg)
    #from model_backbone_unet_hm import build_system
    lit = build_system(cfg)
    # ---------------- Trainer/Logger ----------------
    trainer_cfg = cfg.get('trainer', {})
    epochs = int(trainer_cfg.get('epochs', 100))
    devices = int(trainer_cfg.get('devices', 1))
    precision_cfg = trainer_cfg.get('precision', 'auto')
    precision = 'bf16-mixed' if (precision_cfg == 'auto' and torch.cuda.is_available()) else precision_cfg

    log_cfg = cfg.get('logging', {})
    save_dir = log_cfg.get('save_dir', 'runs')
    exp_name = log_cfg.get('experiment_name', 'default')
    monitor = log_cfg.get('monitor', 'val/dice')
    mode = log_cfg.get('mode', 'min')

    arch = cfg.get("architecture", "unknown")
    # главное: name включает архитектуру → runs/<exp>/<arch>/version_x/…
    logger = TensorBoardLogger(
        save_dir=save_dir,
        name=f"{exp_name}/{arch}"
    )

    # ВАЖНО: не задаём dirpath вручную — тогда PL положит чекпойнты в logger.log_dir/checkpoints
    ckpt_cb = ModelCheckpoint(
        filename="ep{epoch:03d}",
        every_n_epochs=1,
        monitor=monitor,
        mode=log_cfg.get('mode', 'min'),
        save_top_k=-1,
        save_last=True,                    # последний тоже сохраняем
        auto_insert_metric_name=False
    )

    # 2️⃣ лучшая по Dice (чем больше, тем лучше)
    ckpt_best_dice = ModelCheckpoint(
        filename="best-dice-{epoch:03d}-{val/dice:.4f}",
        monitor="val/dice",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False
    )

    # 3️⃣ лучшая по pt_err_px (чем меньше, тем лучше)
    ckpt_best_pts = ModelCheckpoint(
        filename="best-ptErr-{epoch:03d}-{val/pt_err_px:.4f}",
        monitor="val/pt_err_px",
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=False
    )

    ckpt_best_balance = ModelCheckpoint(
        filename="best-balance-{epoch:03d}-{val_balance_score:.4f}",
        monitor="val_balance_score",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False
    )

    run_version = log_cfg.get('version') or datetime.now().strftime("%Y%m%d_%H%M%S")

    tb_logger  = TensorBoardLogger(save_dir=save_dir, name=f"{exp_name}/{arch}", version=run_version)
    csv_logger = CSVLogger(
        save_dir=save_dir,
        name=f"{exp_name}/{arch}",
        version=run_version
    )

    trainer = pl.Trainer(
        default_root_dir=save_dir,         # базовая папка для артефактов
        max_epochs=epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=devices,
        precision=precision,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        logger=[tb_logger, csv_logger],
        callbacks=[ckpt_best_balance],
        enable_progress_bar=True,
        log_every_n_steps=10
    )

    setattr(lit, "log_dir_override", tb_logger.log_dir)

    # инициализация датасета
    dm.setup("fit")

    # базовая статистика (у тебя уже есть метод print_stats())
    dm.print_stats()

    # глубже посмотрим на первые примеры обоих сплитов
    dm.inspect_n_samples(n=4, split="train", outdir="debug_inspect/train")
    dm.inspect_n_samples(n=4, split="val",   outdir="debug_inspect/val")

    # ---------------- Fit ----------------
    trainer.fit(lit, dm)

    # ---------------- best.pt рядом с логами ----------------
    # best_model_path вида: runs/<exp>/<arch>/version_x/checkpoints/epoch=...-val...
    best_path = ckpt_best_balance.best_model_path
    run_dir = Path(tb_logger.log_dir)  
    if best_path:
        src = Path(best_path)
        dst = run_dir / "best.pt"
        try:
            # симлинк удобнее, но не везде доступен → пытаемся, иначе копируем
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src.resolve())
        except Exception:
            shutil.copy2(src, dst)
        print(f"[INFO] Best checkpoint: {src}")
        print(f"[INFO] best.pt placed at: {dst}")
    else:
        print("[WARN] No best_model_path — nothing to link/copy.")

if __name__ == "__main__":
    main()
