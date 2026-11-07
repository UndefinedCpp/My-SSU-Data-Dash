"""Scatterplot of log actual vs log predicted price on the four-month validation set."""
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-config")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "reports/fulltest_error_analysis_predictions.csv"
OUT = REPO / "reports" / "fulltest_log_price_scatter"

BLUE = "#4C72B0"
ORANGE = "#DD8452"
DARK = "#1F2D3D"
MUTED = "#6B7280"
GRID = "#D9DEE5"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#B8C0CC",
    "axes.labelcolor": "#4A5568",
    "xtick.color": "#4A5568",
    "ytick.color": "#4A5568",
})


def main() -> None:
    frame = pd.read_csv(DATA)
    actual = frame["sold_price"].to_numpy(float)
    x = np.log(actual)

    panels = (
        ("Off-market", "off_pred", BLUE),
        ("List-aware", "list_pred", ORANGE),
    )

    low, high = 12.15, 16.85
    ticks = [300_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000]
    tick_labels = ["$300k", "$500k", "$1M", "$2M", "$5M", "$10M"]
    tick_pos = [np.log(v) for v in ticks]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.4), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")

    for ax, (label, column, color) in zip(axes, panels):
        ax.set_facecolor("white")
        y = np.log(frame[column].to_numpy(float))
        ax.scatter(
            x, y, s=9, color=color, alpha=0.22, linewidths=0,
            rasterized=True, zorder=2,
        )
        diagonal = np.linspace(low, high, 200)
        ax.plot(
            diagonal, diagonal, color="#4A5568", linewidth=1.1,
            linestyle=(0, (5, 4)), zorder=3,
        )

        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_labels)
        ax.grid(color=GRID, linewidth=0.7, alpha=0.9)
        ax.set_axisbelow(True)

        ax.set_title(
            f"{label} · n={len(frame):,}",
            loc="left", fontsize=11.5, fontweight="bold", color=DARK, pad=10,
        )
        ax.set_xlabel("Actual sale price (log scale)", labelpad=8)

    axes[0].set_ylabel("Predicted sale price (log scale)", labelpad=8)
    axes[1].tick_params(labelleft=False)

    fig.subplots_adjust(left=0.10, right=0.985, top=0.90, bottom=0.10, wspace=0.08)

    fig.savefig(OUT.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT.with_suffix('.png')} and {OUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
