"""
Ablation study summary figure for BAMNet.
Modern two-panel dot plot showing the impact of each architectural
component on segmentation quality (Dice) and landmark localization
accuracy (Mean / Median Error).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR.parent / "figures"

# ---------- data from publication/evaluation/table_metrics.csv ----------
variants = [
    "Full BAMNet",
    "w/o Coordinate Attention",
    "w/o Deep Feature Fusion",
    "w/o Boundary Loss",
    "w/o Boundary Guidance",
    "Fixed temperature (beta = 8)",
    "w/o Position Attention",
    "Temperature schedule: 4 to 8",
]

# Dice scores
mean_dice = np.array([0.907, 0.902, 0.905, 0.907, 0.900, 0.912, 0.910, 0.910])
med_dice  = np.array([0.9165, 0.9166, 0.9114, 0.9134, 0.9041, 0.9185, 0.9188, 0.9138])

# Localization errors (px)
mean_e = np.array([10.30, 10.62, 10.84, 10.96, 11.04, 10.88, 11.29, 11.37])
med_e  = np.array([8.17, 8.47, 9.01, 8.82, 9.21, 8.77, 9.51, 9.46])

n = len(variants)
y = np.arange(n)

# ---------- palette ----------
BG       = "#FFFFFF"
TEXT     = "#2D2D2D"
TEXT_SEC = "#6B7280"
ACCENT   = "#10B981"  # emerald for full model
DOT_MEAN = "#3B82F6"  # blue — mean values
DOT_MED  = "#F59E0B"  # amber — median values
DOT_ERR_MEAN = "#3B82F6"  # blue — same as DOT_MEAN for consistency
LINE_COL = "#E5E7EB"
STRIP_BG = "#F3F4F6"

# ---------- style ----------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0,
    "xtick.major.width": 0.5,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.minor.size": 0,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT_SEC,
    "ytick.color": TEXT,
})

fig, (ax_dice, ax_err) = plt.subplots(
    1, 2, figsize=(9.0, 4.2), sharey=True,
    gridspec_kw={"width_ratios": [1, 1.4], "wspace": 0.06},
)
fig.patch.set_facecolor(BG)

for ax in (ax_dice, ax_err):
    ax.set_facecolor(BG)

# ---- zebra striping ----
for i in range(n):
    if i % 2 == 0:
        for ax in (ax_dice, ax_err):
            ax.axhspan(i - 0.5, i + 0.5, color=STRIP_BG, zorder=0)

# ---- highlight Full model row ----
for ax in (ax_dice, ax_err):
    ax.axhspan(-0.5, 0.5, color=ACCENT, alpha=0.10, zorder=0)

# ===== LEFT PANEL: Dice Score (mean + median) =====
ax_dice.axvline(mean_dice[0], color=ACCENT, linewidth=0.8, linestyle="--", alpha=0.5, zorder=1)

for i in range(n):
    col_mean = ACCENT if i == 0 else DOT_MEAN
    col_med = ACCENT if i == 0 else DOT_MED

    # connecting line between mean and median
    color_line = ACCENT if i == 0 else "#D1D5DB"
    lw = 1.8 if i == 0 else 1.2
    ax_dice.plot([mean_dice[i], med_dice[i]], [y[i], y[i]],
                 color=color_line, linewidth=lw, solid_capstyle="round", zorder=2)

    # mean dot (square)
    ax_dice.scatter(mean_dice[i], y[i], s=50, color=col_mean, zorder=3,
                    marker="s", edgecolors="white", linewidths=0.8)
    # median dot (circle)
    ax_dice.scatter(med_dice[i], y[i], s=30, color=col_med, zorder=3,
                    marker="o", edgecolors="white", linewidths=0.6)

    # value labels: mean label to the left, median label to the right
    ax_dice.text(mean_dice[i] - 0.0015, y[i], f"{mean_dice[i]:.3f}",
                 va="center", ha="right", fontsize=7.5,
                 color=col_mean, fontweight="bold" if i == 0 else "normal")
    ax_dice.text(med_dice[i] + 0.0015, y[i], f"{med_dice[i]:.3f}",
                 va="center", ha="left", fontsize=7.5,
                 color=col_med, fontweight="bold" if i == 0 else "normal")

ax_dice.set_xlim(0.895, 0.922)
ax_dice.set_xlabel("Dice Score (higher is better)", fontsize=9, fontweight="bold", color=TEXT)
ax_dice.set_yticks(y)
ax_dice.set_yticklabels(variants, fontsize=8.5)
ax_dice.invert_yaxis()

labels = ax_dice.get_yticklabels()
labels[0].set_fontweight("bold")
labels[0].set_color(ACCENT)

ax_dice.xaxis.grid(True, color=LINE_COL, linewidth=0.4, zorder=0)
ax_dice.yaxis.grid(False)
for spine in ax_dice.spines.values():
    spine.set_visible(False)

# ===== RIGHT PANEL: Localization Error (mean + median) =====
ax_err.axvline(mean_e[0], color=ACCENT, linewidth=0.8, linestyle="--", alpha=0.5, zorder=1)

for i in range(n):
    color_line = ACCENT if i == 0 else "#D1D5DB"
    lw = 1.8 if i == 0 else 1.2
    ax_err.plot([med_e[i], mean_e[i]], [y[i], y[i]],
                color=color_line, linewidth=lw, solid_capstyle="round", zorder=2)

    col_med = ACCENT if i == 0 else DOT_MED
    ax_err.scatter(med_e[i], y[i], s=30, color=col_med,
                   marker="o", zorder=3, edgecolors="white", linewidths=0.6)

    col_mean = ACCENT if i == 0 else DOT_ERR_MEAN
    ax_err.scatter(mean_e[i], y[i], s=50, color=col_mean, marker="s",
                   zorder=3, edgecolors="white", linewidths=0.8)

    ax_err.text(mean_e[i] + 0.18, y[i], f"{mean_e[i]:.1f}",
                va="center", ha="left", fontsize=7.5,
                color=col_mean, fontweight="bold" if i == 0 else "normal")
    ax_err.text(med_e[i] - 0.18, y[i], f"{med_e[i]:.1f}",
                va="center", ha="right", fontsize=7.5,
                color=col_med, fontweight="bold" if i == 0 else "normal")

ax_err.set_xlim(7.2, 12.2)
ax_err.set_xlabel("Localization Error, px (lower is better)", fontsize=9, fontweight="bold", color=TEXT)
ax_err.xaxis.grid(True, color=LINE_COL, linewidth=0.4, zorder=0)
ax_err.yaxis.grid(False)
for spine in ax_err.spines.values():
    spine.set_visible(False)

# ===== Legend =====
legend_elements = [
    # mpatches.Patch(facecolor=ACCENT, alpha=0.3, edgecolor=ACCENT,
    #                linewidth=0.8, label="Full BAMNet"),
    plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=DOT_MEAN,
               markersize=7, label="Mean"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=DOT_MED,
               markersize=6, label="Median"),
]
leg = ax_err.legend(handles=legend_elements, loc="upper right", fontsize=7.5,
                    frameon=True, fancybox=True, framealpha=0.95,
                    edgecolor=LINE_COL, handletextpad=0.5, borderpad=0.6,
                    bbox_to_anchor=(1.02, 1.0))
leg.get_frame().set_linewidth(0.5)

fig.tight_layout(pad=0.8)

# ---- save ----
out = FIGURES_DIR / "figure_4" / "ablation_summary.png"
fig.savefig(out, dpi=400, bbox_inches="tight", facecolor=BG)
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor=BG)
print(f"Saved: {out}")
print(f"Saved: {out.with_suffix('.pdf')}")
plt.close()
