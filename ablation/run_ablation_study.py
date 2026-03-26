#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bamnet_paths import expand_config_tree

DEFAULT_ABLATION_SAVE_DIR = "${BAMNET_DATA_ROOT}/ablation_runs"
ACTIVE_VARIANTS = (
    "full_model",
    "no_position_attention",
    "no_coordinate_attention",
    "no_fusion",
    "no_boundary_guidance",
    "no_boundary_loss",
    "beta_fixed_8",
    "beta_schedule_4_8",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate/run BAMNet v4 ablation experiments and collect summary metrics."
    )
    parser.add_argument("--base-config", type=Path, default=ROOT_DIR / "config.yaml")
    parser.add_argument("--train-script", type=Path, default=ROOT_DIR / "train.py")
    parser.add_argument("--python-bin", type=str, default="python3")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument(
        "--save-dir",
        type=str,
        default=DEFAULT_ABLATION_SAVE_DIR,
        help="Override logging.save_dir for generated ablation configs.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="all",
        help="Comma-separated variant names or 'all'.",
    )
    parser.add_argument(
        "--exp-prefix",
        type=str,
        default="ablation_v4",
        help="Prefix for logging.experiment_name (final is <exp-prefix>_<variant>).",
    )
    parser.add_argument("--run", action="store_true", help="Execute training runs sequentially.")
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Skip generation/run and only collect metrics from existing runs.",
    )
    parser.add_argument(
        "--extra-args",
        type=str,
        default="",
        help="Extra CLI args appended to every train command, e.g. '--data_path /data --lr 1e-4'.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Path to output summary CSV. Default: <output-dir>/<exp-prefix>_summary.csv",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def materialize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return expand_config_tree(copy.deepcopy(cfg))


def set_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur: dict[str, Any] = cfg
    for key in parts[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[parts[-1]] = value


def get_nested(cfg: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    parts = dotted_key.split(".")
    cur: Any = cfg
    for key in parts:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    try:
        return float(text)
    except Exception:
        return None


def parse_int(value: str | None) -> int | None:
    parsed = parse_float(value)
    if parsed is None:
        return None
    return int(parsed)


def first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def get_variant_overrides() -> dict[str, dict[str, Any]]:
    return {
        "full_model": {},
        "no_position_attention": {
            "attention.decoder_enabled": False,
            "attention.decoder_attn_levels": 0,
        },
        "no_coordinate_attention": {
            "attention.coordinate": False,
        },
        "no_fusion": {
            "fusion.enabled": False,
        },
        "no_boundary_guidance": {
            "boundary.guidance_enabled": False,
        },
        "no_boundary_loss": {
            "boundary.loss_enabled": False,
            "loss.w_bnd": 0.0,
        },
        "beta_fixed_8": {
            "softargmax.beta_mode": "fixed",
            "softargmax.beta_fixed": 8.0,
        },
        "beta_schedule_4_8": {
            "softargmax.beta_mode": "schedule",
            "softargmax.beta_start": 4.0,
            "softargmax.beta_end": 8.0,
            "softargmax.beta_warmup_epochs": 8,
        },
    }


def select_variants(all_names: list[str], variants_arg: str) -> list[str]:
    if variants_arg.strip().lower() == "all":
        return all_names
    requested = [x.strip() for x in variants_arg.split(",") if x.strip()]
    unknown = [x for x in requested if x not in all_names]
    if unknown:
        raise ValueError(f"[ERR] Unknown variants: {unknown}. Available: {all_names}")
    return requested


def build_variant_config(
    base_cfg: dict[str, Any],
    variant: str,
    overrides: dict[str, Any],
    exp_prefix: str,
    save_dir: str | None,
) -> tuple[dict[str, Any], str]:
    cfg = copy.deepcopy(base_cfg)
    for dotted_key, value in overrides.items():
        set_nested(cfg, dotted_key, value)

    exp_name = f"{exp_prefix}_{variant}"
    set_nested(cfg, "logging.experiment_name", exp_name)
    if save_dir:
        set_nested(cfg, "logging.save_dir", save_dir)
    return cfg, exp_name


def split_extra_args(extra_args: str) -> list[str]:
    text = extra_args.strip()
    if text == "":
        return []
    return shlex.split(text)


def resolve_relative_config_paths(cfg: dict[str, Any], base_dir: Path) -> None:
    for dotted_key in ["data_path", "logging.save_dir"]:
        value = get_nested(cfg, dotted_key)
        if not isinstance(value, str) or value.strip() == "":
            continue
        if "$" in value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (base_dir / path).resolve()
            set_nested(cfg, dotted_key, str(path))


def make_command_path(path: Path, base_dir: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), base_dir.resolve())
    except Exception:
        return str(path.resolve())


def find_latest_metrics_file(save_dir: Path, exp_name: str, architecture: str) -> tuple[Path | None, Path | None]:
    exp_root = save_dir / exp_name / architecture
    if not exp_root.is_dir():
        return None, None

    versions = [p for p in exp_root.iterdir() if p.is_dir()]
    if not versions:
        return None, None

    versions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest_run_dir = versions[0]
    for run_dir in versions:
        metrics_path = run_dir / "metrics.csv"
        if metrics_path.is_file():
            return run_dir, metrics_path
    return latest_run_dir, None


def select_best_metric_key(rows: list[dict[str, str]]) -> str | None:
    for key in ("val_balance_score", "val/score", "val/dice"):
        if any(parse_float(row.get(key)) is not None for row in rows):
            return key
    return None


def summarize_metrics(metrics_path: Path) -> dict[str, Any]:
    with metrics_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return {"status": "empty_metrics"}

    target_key = select_best_metric_key(rows)
    if target_key is None:
        return {"status": "no_validation_rows"}

    best_row: dict[str, str] | None = None
    best_val: float | None = None
    for row in rows:
        val = parse_float(row.get(target_key))
        if val is None:
            continue
        if best_val is None or val > best_val:
            best_val = val
            best_row = row

    if best_row is None:
        return {"status": "no_validation_rows"}

    summary = {
        "status": "ok",
        "best_metric_key": target_key,
        "best_metric_value": parse_float(best_row.get(target_key)),
        "best_epoch": parse_int(best_row.get("epoch")),
        "mean_dice": first_non_none(parse_float(best_row.get("val/dice")), parse_float(best_row.get("val_dice_epoch"))),
        "global_iou": parse_float(best_row.get("val/global_iou")),
        "surface_dice_4px": parse_float(best_row.get("val/surface_dice_4px")),
        "mean_err_px": parse_float(best_row.get("val_pt_err_px")),
        "median_err_px": parse_float(best_row.get("val/median_err_px")),
        "pck_10": parse_float(best_row.get("val/pck_10")),
        "pck_5": parse_float(best_row.get("val/pck_5")),
        "pck_2": parse_float(best_row.get("val/pck_2")),
        "fps": first_non_none(parse_float(best_row.get("val/fps")), parse_float(best_row.get("val/iter_fps"))),
        "latency_ms": parse_float(best_row.get("val/latency_ms")),
        "gpu_mem_mb": first_non_none(parse_float(best_row.get("val/gpu_mem_mb")), parse_float(best_row.get("val/gpu_peak_alloc_mb"))),
        "params_m": first_non_none(parse_float(best_row.get("val/params_m")), parse_float(best_row.get("params_m"))),
        "val_dice": parse_float(best_row.get("val/dice")),
        "val_pt_err_px": parse_float(best_row.get("val_pt_err_px")),
        "val_balance_score": parse_float(best_row.get("val_balance_score")),
        "val_score": parse_float(best_row.get("val/score")),
        "val_iter_fps": parse_float(best_row.get("val/iter_fps")),
        "val_iter_time_s": parse_float(best_row.get("val/iter_time_s")),
        "val_gpu_peak_alloc_mb": parse_float(best_row.get("val/gpu_peak_alloc_mb")),
        "val_gpu_peak_reserved_mb": parse_float(best_row.get("val/gpu_peak_reserved_mb")),
        "train_iter_fps": parse_float(best_row.get("train/iter_fps")),
    }
    return summary


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "status",
        "config_path",
        "run_dir",
        "metrics_csv",
        "metrics_snapshot",
        "best_metric_key",
        "best_metric_value",
        "best_epoch",
        "mean_dice",
        "global_iou",
        "surface_dice_4px",
        "mean_err_px",
        "median_err_px",
        "pck_10",
        "pck_5",
        "pck_2",
        "fps",
        "latency_ms",
        "gpu_mem_mb",
        "params_m",
        "val_dice",
        "val_pt_err_px",
        "val_balance_score",
        "val_score",
        "val_iter_fps",
        "val_iter_time_s",
        "val_gpu_peak_alloc_mb",
        "val_gpu_peak_reserved_mb",
        "train_iter_fps",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_summary_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def export_metrics_snapshot(metrics_path: Path, export_dir: Path, exp_prefix: str, variant: str) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = export_dir / f"{exp_prefix}_{variant}_metrics.csv"
    shutil.copy2(metrics_path, snapshot_path)
    return snapshot_path


def main() -> None:
    args = parse_args()
    base_config_path = args.base_config.expanduser().resolve()
    train_script_path = args.train_script.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    summary_csv = (
        args.summary_csv.expanduser().resolve()
        if args.summary_csv
        else (output_dir / f"{args.exp_prefix}_summary.csv")
    )
    summary_json = summary_csv.with_suffix(".json")
    cfg_dir = output_dir / "configs" / args.exp_prefix
    cmd_file = output_dir / f"{args.exp_prefix}_commands.sh"
    metrics_export_dir = output_dir / "metrics"

    all_overrides = get_variant_overrides()
    variant_names = select_variants(list(all_overrides.keys()), args.variants)
    commands: list[tuple[str, list[str], Path, str, dict[str, Any]]] = []

    base_cfg: dict[str, Any] | None = None
    if base_config_path.is_file():
        base_cfg = load_yaml(base_config_path)

    if not args.collect_only:
        if base_cfg is None:
            raise FileNotFoundError(f"[ERR] Base config not found: {base_config_path}")

        cmd_file.parent.mkdir(parents=True, exist_ok=True)
        cmd_lines: list[str] = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            'cd "$(dirname "$0")"',
            "",
        ]

        for variant in variant_names:
            cfg_variant, exp_name = build_variant_config(
                base_cfg=base_cfg,
                variant=variant,
                overrides=all_overrides[variant],
                exp_prefix=args.exp_prefix,
                save_dir=args.save_dir,
            )
            resolve_relative_config_paths(cfg_variant, output_dir)
            cfg_path = cfg_dir / f"{variant}.yaml"
            dump_yaml(cfg_path, cfg_variant)

            train_cmd_path = make_command_path(train_script_path, output_dir)
            cfg_cmd_path = make_command_path(cfg_path, output_dir)
            cmd = [args.python_bin, train_cmd_path, "--config", cfg_cmd_path, "--exp_name", exp_name]
            cmd += split_extra_args(args.extra_args)
            commands.append((variant, cmd, cfg_path, exp_name, cfg_variant))
            cmd_lines.append(shlex.join(cmd))

        cmd_file.write_text("\n".join(cmd_lines) + "\n", encoding="utf-8")
        print(f"[INFO] Generated {len(commands)} configs in {cfg_dir}")
        print(f"[INFO] Commands saved to {cmd_file}")

        if args.run:
            for variant, cmd, _, _, _ in commands:
                print(f"[INFO] Running {variant}: {' '.join(cmd)}")
                subprocess.run(cmd, check=True, cwd=output_dir)

    summary_rows: list[dict[str, Any]] = []
    for variant in variant_names:
        cfg_path = cfg_dir / f"{variant}.yaml"

        cfg_variant: dict[str, Any] | None = None
        if cfg_path.is_file():
            cfg_variant = load_yaml(cfg_path)
        elif base_cfg is not None:
            cfg_variant, _ = build_variant_config(
                base_cfg=base_cfg,
                variant=variant,
                overrides=all_overrides[variant],
                exp_prefix=args.exp_prefix,
                save_dir=args.save_dir,
            )
            resolve_relative_config_paths(cfg_variant, output_dir)

        if cfg_variant is None:
            summary_rows.append(
                {
                    "variant": variant,
                    "status": "missing_config",
                    "config_path": str(cfg_path),
                    "run_dir": "",
                    "metrics_csv": "",
                    "metrics_snapshot": "",
                }
            )
            continue

        cfg_runtime = materialize_config(cfg_variant)
        exp_name = str(get_nested(cfg_runtime, "logging.experiment_name", f"{args.exp_prefix}_{variant}"))
        save_dir = Path(str(get_nested(cfg_runtime, "logging.save_dir", "runs"))).expanduser().resolve()
        architecture = str(get_nested(cfg_runtime, "architecture", "unknown"))
        run_dir, metrics_path = find_latest_metrics_file(save_dir=save_dir, exp_name=exp_name, architecture=architecture)

        if metrics_path is None:
            summary_rows.append(
                {
                    "variant": variant,
                    "status": "metrics_not_found",
                    "config_path": str(cfg_path),
                    "run_dir": str(run_dir) if run_dir else "",
                    "metrics_csv": "",
                    "metrics_snapshot": "",
                }
            )
            continue

        snapshot_path = export_metrics_snapshot(metrics_path, metrics_export_dir, args.exp_prefix, variant)
        metrics_summary = summarize_metrics(metrics_path)
        row = {
            "variant": variant,
            "config_path": str(cfg_path),
            "run_dir": str(run_dir) if run_dir else "",
            "metrics_csv": str(metrics_path),
            "metrics_snapshot": str(snapshot_path),
        }
        row.update(metrics_summary)
        summary_rows.append(row)

    write_summary_csv(summary_csv, summary_rows)
    write_summary_json(summary_json, summary_rows)
    print(f"[INFO] Summary saved to {summary_csv}")
    print(f"[INFO] Summary saved to {summary_json}")

    ok_count = sum(1 for r in summary_rows if r.get("status") == "ok")
    print(f"[INFO] Collected metrics: {ok_count}/{len(summary_rows)} variants")


if __name__ == "__main__":
    main()
