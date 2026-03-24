#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_METRICS_DIR = SCRIPT_DIR / "metrics"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "publication" / "figures"
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
NAME_MAP = {
    "full_model": "Full Model (BAMNet v4)",
    "no_position_attention": "No PositionAttention",
    "no_coordinate_attention": "No CoordinateAttention",
    "no_fusion": "No Feature Fusion",
    "no_boundary_guidance": "No Boundary Guidance",
    "no_boundary_loss": "No Boundary Loss",
    "beta_fixed_8": "Soft-argmax: fixed 8",
    "beta_schedule_4_8": "Soft-argmax: schedule 4->8",
}
SUMMARY_NAME_MAP = {
    "full_model": "Full model",
    "no_position_attention": "No Pos Attn",
    "no_coordinate_attention": "No Coord Attn",
    "no_fusion": "No Fusion",
    "no_boundary_guidance": "No Bnd Guid",
    "no_boundary_loss": "No Bnd Loss",
    "beta_fixed_8": "Fixed 8",
    "beta_schedule_4_8": "Sched 4->8",
}


# Общий стиль графиков
sns.set(style="whitegrid")
plt.rcParams.update(
    {
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 11,
    }
)

PALETTE = sns.color_palette("muted", n_colors=10)
COLORS = list(PALETTE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot BAMNet v4 ablation results.")
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--exp-prefix",
        type=str,
        default="ablation_v4",
        help="Metrics filename prefix used by run_ablation_study.py.",
    )
    parser.add_argument(
        "--include-comparison",
        action="store_true",
        help="Also render legacy comparison bar charts.",
    )
    return parser.parse_args()


def _save_current_figure(output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")


def _style_axis(ax, xlabel: str = "", ylabel: str = "", y_locator=None, x_locator=None) -> None:
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    if y_locator is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_locator))
    if x_locator is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_locator))
    sns.despine(ax=ax)


def _strip_metrics_suffix(stem: str) -> str:
    if stem.endswith("_metrics"):
        return stem[: -len("_metrics")]
    if stem.endswith("metrics"):
        return stem[: -len("metrics")]
    return stem


def _extract_variant_name(path: Path, exp_prefix: str) -> str | None:
    stem = _strip_metrics_suffix(path.stem)
    normalized_prefix = f"{exp_prefix}_"
    if stem.startswith(normalized_prefix):
        variant = stem[len(normalized_prefix) :]
    else:
        # Legacy tolerance for older exported files.
        variant = stem
        for legacy_prefix in ("ablation_v4_", "ablation_"):
            if variant.startswith(legacy_prefix):
                variant = variant[len(legacy_prefix) :]
                break
    return variant if variant in ACTIVE_VARIANTS else None


def _collect_metric_frames(metrics_dir: Path, exp_prefix: str) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for csv_path in sorted(metrics_dir.glob("*.csv")):
        variant = _extract_variant_name(csv_path, exp_prefix)
        if variant is None:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"Error processing {csv_path}: {exc}")
            continue
        if "val/dice" not in df.columns or "val_pt_err_px" not in df.columns:
            print(f"Skipping {csv_path}: required columns are missing.")
            continue
        frames.append(
            {
                "variant": variant,
                "display_name": NAME_MAP[variant],
                "summary_name": SUMMARY_NAME_MAP[variant],
                "path": csv_path,
                "df": df,
            }
        )

    ordered_frames: list[dict[str, object]] = []
    by_variant = {frame["variant"]: frame for frame in frames}
    for variant in ACTIVE_VARIANTS:
        frame = by_variant.get(variant)
        if frame is not None:
            ordered_frames.append(frame)
    return ordered_frames


def _balance_score_from_series(df: pd.DataFrame) -> pd.Series:
    pt_score = 1.0 / (1.0 + (df["val_pt_err_px"] / 10.0))
    return 0.6 * df["val/dice"] + 0.4 * pt_score


def _select_best_summary_row(df: pd.DataFrame) -> pd.Series:
    required_cols = ["val/dice", "val_pt_err_px"]
    valid_df = df.dropna(subset=required_cols).copy()
    if valid_df.empty:
        raise ValueError("No rows with both val/dice and val_pt_err_px available.")

    if "val_balance_score" in valid_df.columns:
        balance_score = valid_df["val_balance_score"].fillna(_balance_score_from_series(valid_df))
    else:
        balance_score = _balance_score_from_series(valid_df)

    best_idx = balance_score.idxmax()
    return valid_df.loc[best_idx]


def plot_ablation_learning_curves(metrics_dir: Path, output_dir: Path, exp_prefix: str) -> None:
    print("Plotting ablation learning curves...")
    frames = _collect_metric_frames(metrics_dir, exp_prefix)
    if not frames:
        print(f"No matching metrics files found in {metrics_dir} for prefix '{exp_prefix}'.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for i, frame in enumerate(frames):
        df = frame["df"]
        display_name = str(frame["display_name"])

        span = 5
        dice_smooth = df["val/dice"].ewm(span=span).mean()
        err_smooth = df["val_pt_err_px"].ewm(span=span).mean()
        epochs = df["epoch"]

        color = COLORS[i % len(COLORS)]
        is_full = frame["variant"] == "full_model"
        linewidth = 3.0 if is_full else 2.0
        alpha = 1.0 if is_full else 0.75

        ax1.plot(epochs, dice_smooth, label=display_name, color=color, linewidth=linewidth, alpha=alpha)
        ax2.plot(epochs, err_smooth, label=display_name, color=color, linewidth=linewidth, alpha=alpha)

    ax1.set_title("Validation Dice Progression")
    _style_axis(ax1, xlabel="Epoch", ylabel="Dice", y_locator=0.02)
    ax1.legend(loc="lower right", frameon=False)

    ax2.set_title("Validation Landmark Error Progression")
    _style_axis(ax2, xlabel="Epoch", ylabel="Error (px)", y_locator=1.0)
    ax2.legend(loc="upper right", frameon=False)

    _save_current_figure(output_dir, "ablation_curves")
    print(f"Saved to {output_dir / 'ablation_curves.pdf'}")


def plot_comparison_bar_charts(output_dir: Path) -> None:
    print("Plotting comparison bar charts...")

    models_seg = ["Swin-Unet", "YOLO-seg (L)", "YOLO-seg (M)", "MaNet", "BAMNet (Ours)"]
    dice_scores = [0.8352, 0.8585, 0.8826, 0.8790, 0.8873]
    iou_scores = [0.7276, 0.7591, 0.7958, 0.8120, 0.8053]

    models_pts = ["YOLO detect (L)", "YOLO detect (M)", "RT-DETR", "YOLO-KP (L)", "YOLO-KP (M)", "BAMNet (Ours)"]
    pt_errors = [5.32, 4.00, 5.94, 9.51, 3.69, 3.16]

    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(models_seg))
    width = 0.35

    ax.bar(x - width / 2, dice_scores, width, label="Dice Score", color=COLORS[0], alpha=0.9, edgecolor="black", linewidth=1.0)
    ax.bar(x + width / 2, iou_scores, width, label="IoU Score", color=COLORS[1], alpha=0.9, edgecolor="black", linewidth=1.0)

    ax.set_title("Segmentation Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models_seg, rotation=20, ha="right")
    ax.legend(loc="lower right", frameon=False)
    ax.set_ylim(0.65, 0.95)
    _style_axis(ax, ylabel="Metric Value", y_locator=0.05)

    _save_current_figure(output_dir, "comparison_segmentation")

    fig, ax = plt.subplots(figsize=(12, 8))
    colors_pts = [COLORS[7]] * (len(models_pts) - 1) + [COLORS[2]]
    x_pts = np.arange(len(models_pts))

    bars = ax.bar(x_pts, pt_errors, color=colors_pts, alpha=0.9, edgecolor="black", linewidth=1.0)
    ax.set_title("Landmark Localization Accuracy (Lower is Better)")
    ax.set_xticks(x_pts)
    ax.set_xticklabels(models_pts, rotation=20, ha="right")
    _style_axis(ax, ylabel="Mean Euclidean Error (mm)", y_locator=1.0)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    _save_current_figure(output_dir, "comparison_landmarks")


def plot_ablation_summary(metrics_dir: Path, output_dir: Path, exp_prefix: str) -> None:
    print("Plotting ablation summary metrics...")
    frames = _collect_metric_frames(metrics_dir, exp_prefix)
    if not frames:
        print(f"No matching metrics files found in {metrics_dir} for prefix '{exp_prefix}'.")
        return

    results = []
    for frame in frames:
        df = frame["df"]
        best_row = _select_best_summary_row(df)
        results.append(
            {
                "Model": frame["summary_name"],
                "Variant": frame["variant"],
                "Dice": float(best_row["val/dice"]),
                "Error": float(best_row["val_pt_err_px"]),
            }
        )

    res_df = pd.DataFrame(results).sort_values(by="Dice", ascending=True)

    fig, ax1 = plt.subplots(figsize=(12, 9))
    y_pos = np.arange(len(res_df))

    colors_summary = [COLORS[7]] * len(res_df)
    bamnet_rows = res_df.index[res_df["Variant"] == "full_model"].tolist()
    for idx in bamnet_rows:
        colors_summary[list(res_df.index).index(idx)] = COLORS[2]

    ax1.barh(
        y_pos,
        res_df["Dice"],
        height=0.65,
        color=colors_summary,
        alpha=0.9,
        edgecolor="black",
        linewidth=1.0,
        label="Dice",
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(res_df["Model"], fontweight="bold")
    ax1.set_title("Ablation Study: Summary Metrics")

    # Use fixed limits for better visual comparison and less exaggeration
    dice_min = 0.88
    dice_max = 0.91
    ax1.set_xlim(dice_min, dice_max)

    ax2 = ax1.twiny()
    ax2.plot(res_df["Error"], y_pos, color=COLORS[3], marker="D", linestyle="", markersize=9, label="Error (px)")
    # Use fixed limits for error
    err_min = 10.0
    err_max = 12.0
    ax2.set_xlim(err_min, err_max)

    _style_axis(ax1, xlabel="Dice Score")
    ax1.xaxis.set_major_locator(MultipleLocator(0.02))
    ax2.set_xlabel("Mean Landmark Error (px)", fontsize=16, color="black")
    ax2.tick_params(axis="x", labelsize=13, colors="black")
    ax2.xaxis.set_major_locator(MultipleLocator(1))
    sns.despine(ax=ax2, left=True, bottom=True)

    _save_current_figure(output_dir, "ablation_summary")


def main() -> None:
    args = parse_args()
    plot_ablation_learning_curves(args.metrics_dir, args.output_dir, args.exp_prefix)
    plot_ablation_summary(args.metrics_dir, args.output_dir, args.exp_prefix)
    if args.include_comparison:
        plot_comparison_bar_charts(args.output_dir)
    print("Plot generation completed.")


if __name__ == "__main__":
    main()
