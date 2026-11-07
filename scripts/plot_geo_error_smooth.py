"""Smoothed geographic signed-error surface for the four-month validation set."""
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-config")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import binary_closing, binary_erosion, gaussian_filter
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic_2d

REPO = Path(__file__).resolve().parents[1]
PRED = REPO / "reports" / "fulltest_error_analysis_predictions.csv"
LISTINGS = REPO / "data" / "listings.parquet"
OUT = REPO / "reports" / "fulltest_geo_signed_error_smooth"

DARK = "#1F2D3D"
MUTED = "#6B7280"
LAT0, LON0 = 43.72, -79.38
KM_PER_DEG = 111.32
COS_LAT = np.cos(np.deg2rad(43.7))


def project(lat, lon):
    return (lon - LON0) * KM_PER_DEG * COS_LAT, (lat - LAT0) * KM_PER_DEG


def smooth_surface(x, y, values, extent, bin_km=1.2, sigma_km=1.8):
    xmin, xmax, ymin, ymax = extent
    xedges = np.arange(xmin, xmax + bin_km, bin_km)
    yedges = np.arange(ymin, ymax + bin_km, bin_km)
    med, _, _, _ = binned_statistic_2d(
        x, y, values, statistic="median", bins=[xedges, yedges]
    )
    cnt, _, _, _ = binned_statistic_2d(
        x, y, values, statistic="count", bins=[xedges, yedges]
    )
    med, cnt = med.T, cnt.T
    xc = (xedges[:-1] + xedges[1:]) / 2
    yc = (yedges[:-1] + yedges[1:]) / 2
    Xc, Yc = np.meshgrid(xc, yc)

    valid = cnt >= 3
    pts = np.column_stack([Xc[valid], Yc[valid]])
    vals = med[valid]

    fx = np.linspace(xmin, xmax, 280)
    fy = np.linspace(ymin, ymax, 240)
    Xf, Yf = np.meshgrid(fx, fy)
    linear = griddata(pts, vals, (Xf, Yf), method="linear")
    nearest = griddata(pts, vals, (Xf, Yf), method="nearest")
    filled = np.where(np.isnan(linear), nearest, linear)

    sigma_cells = sigma_km / (fx[1] - fx[0])
    smoothed = gaussian_filter(filled, sigma=sigma_cells, mode="nearest")

    tree = cKDTree(np.column_stack([x, y]))
    dist, _ = tree.query(np.column_stack([Xf.ravel(), Yf.ravel()]), k=8)
    support = (dist[:, -1] <= 2.2).reshape(Xf.shape)
    support = binary_closing(support, structure=np.ones((5, 5)))
    support = gaussian_filter(support.astype(float), sigma=2.0) > 0.5
    support = binary_erosion(support, structure=np.ones((5, 5)))
    return Xf, Yf, np.ma.array(smoothed, mask=~support)


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

    levels = np.linspace(-12, 12, 81)
    panels = (
        (axs[0], "off_signed", "Off-market · median signed error (%)"),
        (axs[1], "list_signed", "List-aware · median signed error (%)"),
    )

    for ax, column, title in panels:
        Xf, Yf, surface = smooth_surface(
            x, y, frame[column].to_numpy(float), extent
        )
        cf = ax.contourf(
            Xf, Yf, surface, levels=levels, cmap=plt.get_cmap("RdBu_r"),
            extend="both", antialiased=True, zorder=2,
        )
        ax.contour(
            Xf, Yf, np.ma.getmaskarray(surface).astype(float),
            levels=[0.5], colors="#CBD5E1", linewidths=0.8, zorder=3,
        )
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

    sm = plt.cm.ScalarMappable(
        norm=plt.Normalize(-12, 12), cmap=plt.get_cmap("RdBu_r")
    )
    cb = fig.colorbar(sm, cax=cax, extend="both")
    cb.set_label("Median signed error (%)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)

    fig.subplots_adjust(left=0.035, right=0.94, top=0.90, bottom=0.035)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT.with_suffix('.png')} and .pdf")


if __name__ == "__main__":
    main()
