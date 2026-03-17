import copy
import os
import sys
from pathlib import Path

import yaml


DEVICE = os.getenv("CUDA_VISIBLE_DEVICES", "0,1")
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE

from ultralytics import YOLO  # noqa: E402
from ultralytics import settings  # noqa: E402


SPLIT_NAMES = ["train", "val"]
FOLD_PREFIX = "fold_"
GENERATED_DATA_CFG_DIR = Path("config/_generated_fold_data")
DRY_RUN = os.getenv("TRAIN_CONF_DRY_RUN", "0") == "1"


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def save_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def resolve_dataset_root(cfg: dict, cfg_path: Path) -> Path | None:
    raw_path = cfg.get("path")
    if raw_path is None:
        return None

    path = Path(str(raw_path))
    if path.is_absolute():
        return path.resolve()
    return (cfg_path.parent / path).resolve()


def is_fold_root(path: Path) -> bool:
    if not path.is_dir() or not path.name.startswith(FOLD_PREFIX):
        return False
    return all((path / "images" / split_name).exists() for split_name in SPLIT_NAMES)


def discover_fold_roots(dataset_root: Path) -> list[Path]:
    if not dataset_root.exists() or not dataset_root.is_dir():
        return []
    return sorted([d.resolve() for d in dataset_root.iterdir() if is_fold_root(d)])


def build_fold_data_cfg(cfg: dict, fold_root: Path) -> dict:
    fold_cfg = copy.deepcopy(cfg)
    fold_cfg["path"] = str(fold_root)
    return fold_cfg


def build_generated_data_cfg_path(cfg_path: Path, fold_name: str) -> Path:
    stem = cfg_path.stem
    return GENERATED_DATA_CFG_DIR / f"{stem}__{fold_name}.yaml"


def build_training_jobs(cfg: dict, cfg_path: Path) -> list[dict]:
    dataset_root = resolve_dataset_root(cfg, cfg_path)
    fold_roots = discover_fold_roots(dataset_root) if dataset_root is not None else []

    if not fold_roots:
        return [
            {
                "fold_name": None,
                "data_arg": str(cfg_path),
                "dataset_root": dataset_root,
            }
        ]

    jobs: list[dict] = []
    for fold_root in fold_roots:
        fold_name = fold_root.name
        fold_cfg = build_fold_data_cfg(cfg, fold_root)
        fold_cfg_path = build_generated_data_cfg_path(cfg_path, fold_name)
        save_yaml(fold_cfg_path, fold_cfg)
        jobs.append(
            {
                "fold_name": fold_name,
                "data_arg": str(fold_cfg_path),
                "dataset_root": fold_root,
            }
        )
    return jobs


def build_run_name(cfg: dict, fold_name: str | None) -> str:
    model_stem = Path(str(cfg["model"])).stem
    base_name = f"{cfg['name']}_{model_stem}"
    if not fold_name:
        return base_name
    return f"{base_name}_{fold_name}"


def train_one_job(cfg: dict, config_path: Path, job: dict) -> None:
    fold_name = job["fold_name"]
    data_arg = job["data_arg"]
    run_name = build_run_name(cfg, fold_name)

    if fold_name:
        print(f"[INFO] Fold: {fold_name}")
        print(f"[INFO] Fold dataset root: {job['dataset_root']}")
        print(f"[INFO] Fold data config: {data_arg}")
    else:
        print(f"[INFO] Data config: {config_path}")

    print(f"[INFO] Run name: {cfg['project']}/{run_name}")

    if DRY_RUN:
        print("[INFO] DRY RUN: training skipped (TRAIN_CONF_DRY_RUN=1)")
        return

    # Важно: новый экземпляр на каждый fold, чтобы веса не продолжали обучение с предыдущего fold.
    model = YOLO(cfg["model"], cfg["task"])
    model.train(
        data=data_arg,
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        project=cfg["project"],
        name=run_name,
        device=DEVICE,
        batch=12,
        workers=8,
        patience=5,
        cos_lr=True,
    )


def main() -> None:
    # mlflow server --host 0.0.0.0 --backend-store-uri runs/mlflow
    conf_list = [
        # "config/config_yolo26m_seg.yaml",
        # "config/config_yolo26l_seg.yaml",
        # "config/config_yolov26m_pose.yaml",
        # "config/config_yolov26l_pose.yaml",
        # "config/config_yolov26m.yaml",
        #"config/config_yolov26l.yaml",
        "config/config_RT-DETR.yaml"
    ]

    settings.update({"mlflow": True})

    for conf_path_str in conf_list:
        cfg_path = Path(conf_path_str).resolve()
        if not cfg_path.exists():
            print(f"[ERROR] Config not found: {cfg_path}")
            sys.exit(1)

        try:
            cfg = load_yaml(cfg_path)
        except Exception as e:
            print(f"[ERROR] Failed to load config {cfg_path}: {e}")
            sys.exit(1)

        required_keys = ["model", "task", "epochs", "imgsz", "project", "name"]
        missing = [key for key in required_keys if key not in cfg]
        if missing:
            print(f"[ERROR] Missing required keys in {cfg_path}: {missing}")
            sys.exit(1)

        print("")
        print(f"[INFO] Training config: {cfg_path}")
        print(f"[INFO] Model: {cfg['model']} | task={cfg['task']}")
        print(f"[INFO] Epochs: {cfg['epochs']} | imgsz={cfg['imgsz']} | device={DEVICE}")

        jobs = build_training_jobs(cfg, cfg_path)
        fold_jobs = [job for job in jobs if job["fold_name"] is not None]
        if fold_jobs:
            print(f"[INFO] Detected {len(fold_jobs)} folds in dataset path: {cfg.get('path')}")
            print("[INFO] Folds: " + ", ".join(job["fold_name"] for job in fold_jobs))
        else:
            print(f"[INFO] Fold mode not detected. Training single dataset from config path: {cfg.get('path')}")

        for job in jobs:
            train_one_job(cfg=cfg, config_path=cfg_path, job=job)


if __name__ == "__main__":
    main()
