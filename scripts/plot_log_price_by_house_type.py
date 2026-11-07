"""Plot the distribution of log sale price by house type."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA = Path("data/listings.parquet")
OUT_DIR = Path("reports")
OUT_STEM = "log_price_by_house_type"

BLUE = "#4C72B0"
DARK = "#1F2D3D"
GRID = "#D9DEE5"
MUTED = "#6B7280"


def main() -> None:
    df = pd.read_parquet(DATA)
    df["log_price"] = np.log(df["sold_price"].astype(float))

    stats = (
        df.groupby("house_type_name")["log_price"]
        .agg(n="size", median="median", q25=lambda s: s.quantile(0.25),
             q75=lambda s: s.quantile(0.75))
        .sort_values("median")
    )
    order = stats.index.tolist()
    data = [df.loc[df["house_type_name"] == name, "log_price"].to_numpy()
            for name in order]
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    parts = ax.violinplot(
        data,
        positions=y,
        orientation="horizontal",
        widths=0.82,
        showextrema=False,
        showmeans=False,
        showmedians=False,
        points=220,
        bw_method=0.22,
    )
    for body in parts["bodies"]:
        body.set_facecolor(BLUE)
        body.set_edgecolor("#2F4B6E")
        body.set_linewidth(0.8)
        body.set_alpha(0.42)

    for yi, name in zip(y, order):
        row = stats.loc[name]
        ax.text(
            1.018,
            yi,
            f"n={int(row['n']):,}",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=9,
            color=MUTED,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=10.5, color=DARK)
    ax.set_ylim(-0.65, len(order) - 0.35)

    low, high = df["log_price"].quantile([0.005, 0.995])
    ax.set_xlim(low - 0.06, high + 0.08)
    ticks = [300_000, 500_000, 750_000, 1_000_000, 1_500_000, 2_000_000, 3_000_000, 5_000_000]
    labels = ["$300k", "$500k", "$750k", "$1M", "$1.5M", "$2M", "$3M", "$5M"]
    ax.set_xticks([np.log(v) for v in ticks])
    ax.set_xticklabels(labels)

    ax.set_xlabel("Sale price (log scale)", fontsize=10, color="#4A5568", labelpad=10)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#B8C0CC")
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", colors="#4A5568", labelsize=9.5, length=3)

    ax.set_title(
        "Sale price distribution by house type",
        loc="left",
        fontsize=16,
        fontweight="bold",
        color=DARK,
        pad=14,
    )

    fig.subplots_adjust(left=0.19, right=0.865, top=0.835, bottom=0.12)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{OUT_STEM}.png", dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{OUT_STEM}.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(stats.round(3).to_string())
    print(f"wrote {OUT_DIR / (OUT_STEM + '.png')} and .pdf")


if __name__ == "__main__":
    main()
