"""Hexbin heatmaps of median signed error by geographic location."""
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-config")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PRED = REPO / "reports" / "fulltest_error_analysis_predictions.csv"
LISTINGS = REPO / "data" / "listings.parquet"
OUT = REPO / "reports" / "fulltest_geo_signed_error_heatmap"

DARK = "#1F2D3D"
MUTED = "#6B7280"
LAT0, LON0 = 43.72, -79.38
KM_PER_DEG = 111.32
COS_LAT = np.cos(np.deg2rad(43.7))


def project(lat, lon):
    x = (lon - LON0) * KM_PER_DEG * COS_LAT
    y = (lat - LAT0) * KM_PER_DEG
    return x, y


def main():
    frame = pd.read_csv(PRED)
    coords = pd.read_parquet(
        LISTINGS, columns=["ml_num", "latitude", "longitude"]
    )
    frame = frame.merge(coords, on="ml_num", how="left", validate="one_to_one")
    if frame[["latitude", "longitude"]].isna().any().any():
        raise ValueError("Missing coordinates after merge")

    x, y = project(frame["latitude"].to_numpy(), frame["longitude"].to_numpy())
    pad = 1.0
    extent = (x.min() - pad, x.max() + pad, y.min() - pad, y.max() + pad)

    fig = plt.figure(figsize=(14.2, 7.4))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.025], wspace=0.035)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[0, 2])

    panels = (
        (axs[0], "off_signed", "Off-market · median signed error (%)"),
        (axs[1], "list_signed", "List-aware · median signed error (%)"),
    )
    norm = plt.Normalize(-12, 12)

    arrays = {}
    for ax, column, title in panels:
        ax.set_facecolor("white")
        ax.scatter(
            x, y, s=2.5, color="#CBD5E1", alpha=0.30, linewidths=0,
            rasterized=True, zorder=1,
        )
        hb = ax.hexbin(
            x, y, C=frame[column].to_numpy(float),
            reduce_C_function=np.median,
            gridsize=30, mincnt=5, extent=extent,
            cmap=plt.get_cmap("RdBu_r"), norm=norm,
            linewidths=0.0, zorder=2,
        )
        offsets = hb.get_offsets()
        values = np.ma.filled(np.ma.asarray(hb.get_array()), np.nan)
        arrays[column] = (offsets[:, 0], offsets[:, 1], values)

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_anchor("N")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold",
                     color=DARK, pad=9)

    # 10 km scale bar in the left panel.
    ax = axs[0]
    bar_y = extent[2] + 1.6
    ax.plot([extent[0] + 1.5, extent[0] + 11.5], [bar_y, bar_y],
            color=DARK, linewidth=2.2, solid_capstyle="butt", zorder=5)
    ax.text(extent[0] + 6.5, bar_y + 0.8, "10 km", ha="center", va="bottom",
            fontsize=8.5, color=DARK, zorder=5,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("RdBu_r"))
    cb = fig.colorbar(sm, cax=cax, extend="both")
    cb.set_label("Median signed error (%)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)

    fig.subplots_adjust(left=0.035, right=0.94, top=0.90, bottom=0.035)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)

    off = arrays["off_signed"]
    li = arrays["list_signed"]
    out = pd.DataFrame({
        "x_km": off[0], "y_km": off[1],
        "off_signed": off[2], "list_signed": li[2],
    }).dropna()
    out.to_csv(REPO / "reports" / "fulltest_geo_signed_error_hexbins.csv", index=False)
    print(f"wrote {OUT.with_suffix('.png')} and .pdf")
    print(f"hex cells: {len(out)}")
    print(out[["off_signed", "list_signed"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
