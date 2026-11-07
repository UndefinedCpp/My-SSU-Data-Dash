"""Three error-analysis figures for the frozen four-month validation window.

Uses the two saved checkpoints trained at the 2026-02-01 cutoff and scores
exactly the harness fulltest rows: homes listed and sold from February through
May 2026.

Outputs (under reports/):
    fulltest_calibration.png / .pdf
    fulltest_error_cdf.png / .pdf
    fulltest_type_price_heatmap.png / .pdf
    fulltest_error_analysis_predictions.csv
    fulltest_type_price_heatmap.csv
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-config")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import NullLocator

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import ape_by_price_decile as base  # noqa: E402
import harness  # noqa: E402

OUT_DIR = REPO / "reports"
MODEL_DIR = REPO / "saved_models/fulltest"
CUTOFF = base.CUTOFF
OFF_ID = "OffMarketRefinedV6"
LIST_ID = "ListAwareTextStackV6Tuned"

BLUE = "#4C72B0"
ORANGE = "#DD8452"
DARK = "#1F2D3D"
MUTED = "#6B7280"
GRID = "#D9DEE5"
BAD = "#E5E7EB"

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


def load_predictions():
    valid = base.validation_frame()
    target = valid["sold_price"].to_numpy(float)
    codes, edges = pd.qcut(
        target, 10, labels=False, retbins=True, duplicates="drop"
    )
    n_deciles = len(edges) - 1

    preds = {}
    for spec in base.MODELS:
        model, metadata, model_path = base.load_checkpoint(MODEL_DIR, spec["id"])
        if model.TRACK != metadata["track"]:
            raise ValueError(f"{spec['id']}: checkpoint track mismatch")
        pred = base.predict(model, valid)
        metrics = harness.report_metrics(pd.Series(pred), pd.Series(target))
        recorded = metadata.get("metrics") or {}
        for key in ("N", "MdAPE", "MAPE"):
            if key in recorded and not np.isclose(
                metrics[key], recorded[key], rtol=0, atol=1e-9
            ):
                raise AssertionError(
                    f"{spec['id']}: reloaded {key}={metrics[key]} != "
                    f"recorded {recorded[key]}"
                )
        preds[spec["id"]] = pred
        print(
            f"{spec['id']}: N={metrics['N']:,} "
            f"MdAPE={metrics['MdAPE']:.4f} MAPE={metrics['MAPE']:.4f}"
        )

    frame = valid[
        ["ml_num", "date_listed", "house_type_name", "community", "sold_price"]
    ].copy()
    frame["price_decile"] = codes + 1
    frame["off_pred"] = preds[OFF_ID]
    frame["list_pred"] = preds[LIST_ID]
    for prefix, model_id in (("off", OFF_ID), ("list", LIST_ID)):
        pred = preds[model_id]
        frame[f"{prefix}_ape"] = 100 * np.abs(pred - target) / target
        frame[f"{prefix}_signed"] = 100 * (pred - target) / target
    frame.to_csv(OUT_DIR / "fulltest_error_analysis_predictions.csv", index=False)

    return valid, target, codes, edges, n_deciles, preds


def draw_calibration(target, codes, n_deciles, preds):
    rows = []
    for index in range(n_deciles):
        mask = codes == index
        row = {
            "price_decile": index + 1,
            "n": int(mask.sum()),
            "median_actual": float(np.median(target[mask])),
        }
        for model_id in (OFF_ID, LIST_ID):
            row[model_id] = float(np.median(preds[model_id][mask]))
        rows.append(row)
    cal = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.4, 6.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    low, high = 350_000, 2_800_000
    diagonal = np.geomspace(low, high, 200)
    ax.fill_between(
        diagonal, diagonal * 0.9, diagonal * 1.1,
        color="#9CA3AF", alpha=0.15, linewidth=0, label="±10% band",
    )
    ax.plot(
        diagonal, diagonal, color="#6B7280", linewidth=1.2,
        linestyle=(0, (5, 4)), label="Perfect calibration",
    )

    for model_id, label, color in (
        (OFF_ID, "Off-market", BLUE),
        (LIST_ID, "List-aware", ORANGE),
    ):
        ax.plot(
            cal["median_actual"], cal[model_id],
            color=color, linewidth=1.8, marker="o", markersize=6.0,
            markeredgecolor="white", markeredgewidth=0.8, label=label,
            zorder=4,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")

    ticks = [400_000, 500_000, 750_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000]
    labels = ["$400k", "$500k", "$750k", "$1M", "$1.5M", "$2M", "$2.5M"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())

    ax.set_xlabel("Median actual sale price", labelpad=9)
    ax.set_ylabel("Median predicted sale price", labelpad=9)
    ax.grid(which="major", color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)

    off_top = cal.iloc[-1]
    list_top = cal.iloc[-1]
    ax.annotate(
        "D10", xy=(off_top["median_actual"], off_top[OFF_ID]),
        xytext=(-46, 4), textcoords="offset points",
        fontsize=8, color=BLUE, fontweight="bold",
    )
    ax.annotate(
        "D10", xy=(list_top["median_actual"], list_top[LIST_ID]),
        xytext=(8, -18), textcoords="offset points",
        fontsize=8, color=ORANGE, fontweight="bold",
    )
    ax.annotate(
        "D1", xy=(cal.iloc[0]["median_actual"], cal.iloc[0][OFF_ID]),
        xytext=(8, -15), textcoords="offset points",
        fontsize=8, color=MUTED,
    )

    ax.set_title(
        "Calibration by price decile",
        loc="left", fontsize=14, fontweight="bold", color=DARK, pad=12,
    )
    ax.legend(
        loc="upper left", frameon=False, fontsize=8.8,
        handlelength=1.8, borderaxespad=0.8,
    )

    cal.to_csv(OUT_DIR / "fulltest_calibration.csv", index=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fulltest_calibration.png", dpi=300, facecolor="white")
    fig.savefig(OUT_DIR / "fulltest_calibration.pdf", facecolor="white")
    plt.close(fig)
    print(cal.round(0).to_string(index=False))


def draw_cdf_from_frame(frame):
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for prefix, label, color in (
        ("off", "Off-market", BLUE),
        ("list", "List-aware", ORANGE),
    ):
        ape = np.sort(frame[f"{prefix}_ape"].to_numpy(float))
        share = np.arange(1, len(ape) + 1) / len(ape) * 100
        x = np.concatenate([[0.0], ape])
        y = np.concatenate([[0.0], share])
        ax.step(x, y, where="post", color=color, linewidth=2.0, label=label)
        p90 = float(np.quantile(ape, 0.90))
        ax.vlines(p90, 0, 90, color=color, linewidth=1.0, linestyle=(0, (3, 3)), alpha=0.8)
        ax.text(
            p90 + 0.8, 91.5, f"90% ≤ {p90:.1f}%",
            color=color, fontsize=8.8, va="bottom", ha="left",
        )

    ax.axhline(90, color="#9CA3AF", linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(0.8, 86.5, "90% of homes", color=MUTED, fontsize=8.2, va="center")

    ax.set_xlim(0, 50)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_yticks([0, 25, 50, 75, 90, 100])
    ax.set_xlabel("Absolute percentage error (%)", labelpad=8)
    ax.set_ylabel("Share of homes within error (%)", labelpad=8)
    ax.grid(color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)

    ax.text(
        49.5, 20.0,
        "Off-market tail: 25 errors above 50% (max 168%)",
        ha="right", va="bottom", fontsize=7.8, color=MUTED,
    )
    ax.set_title(
        "Share of homes within a given error",
        loc="left", fontsize=14, fontweight="bold", color=DARK, pad=12,
    )
    ax.legend(
        loc="lower right", frameon=False, fontsize=9,
        handlelength=2.0, borderaxespad=1.0,
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fulltest_error_cdf.png", dpi=300, facecolor="white")
    fig.savefig(OUT_DIR / "fulltest_error_cdf.pdf", facecolor="white")
    plt.close(fig)


def draw_heatmap(frame, codes, n_deciles):
    types = (
        frame.groupby("house_type_name")["sold_price"]
        .median()
        .sort_values()
        .index.tolist()
    )
    n = np.zeros((len(types), n_deciles), dtype=int)
    md_off = np.full((len(types), n_deciles), np.nan)
    md_list = np.full((len(types), n_deciles), np.nan)
    sg_off = np.full((len(types), n_deciles), np.nan)
    sg_list = np.full((len(types), n_deciles), np.nan)

    type_values = frame["house_type_name"].to_numpy()
    for i, name in enumerate(types):
        for j in range(n_deciles):
            mask = (type_values == name) & (codes == j)
            count = int(mask.sum())
            n[i, j] = count
            if count < 20:
                continue
            md_off[i, j] = float(np.median(frame.loc[mask, "off_ape"]))
            md_list[i, j] = float(np.median(frame.loc[mask, "list_ape"]))
            sg_off[i, j] = float(np.median(frame.loc[mask, "off_signed"]))
            sg_list[i, j] = float(np.median(frame.loc[mask, "list_signed"]))

    rows = []
    for i, name in enumerate(types):
        for j in range(n_deciles):
            rows.append({
                "house_type": name,
                "price_decile": j + 1,
                "n": int(n[i, j]),
                "off_mdape": md_off[i, j],
                "list_mdape": md_list[i, j],
                "off_signed": sg_off[i, j],
                "list_signed": sg_list[i, j],
            })
    pd.DataFrame(rows).to_csv(OUT_DIR / "fulltest_type_price_heatmap.csv", index=False)

    fig = plt.figure(figsize=(15.0, 9.6))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        2, 3, width_ratios=[1, 1, 0.035], height_ratios=[1, 1],
        wspace=0.06, hspace=0.22,
    )
    axs = np.array([
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
    ])
    caxes = [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])]

    def panel(ax, data, cmap, norm, title, show_y, show_x):
        cmap = cmap.copy()
        cmap.set_bad(BAD)
        im = ax.imshow(
            data, origin="lower", aspect="auto", interpolation="nearest",
            cmap=cmap, norm=norm, extent=(-0.5, n_deciles - 0.5, -0.5, len(types) - 0.5),
        )
        ax.set_xlim(-0.5, n_deciles - 0.5)
        ax.set_ylim(-0.5, len(types) - 0.5)
        ax.set_xticks(np.arange(-0.5, n_deciles, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(types), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.4)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xticks(range(n_deciles))
        ax.set_xticklabels([f"D{j + 1}" for j in range(n_deciles)], fontsize=7.8)
        ax.set_yticks(range(len(types)))
        if show_y:
            ax.set_yticklabels(types, fontsize=9)
        else:
            ax.tick_params(labelleft=False)
        if not show_x:
            ax.tick_params(labelbottom=False)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=DARK, pad=7)

        for i in range(len(types)):
            for j in range(n_deciles):
                value = data[i, j]
                if np.isnan(value):
                    ax.text(j, i, "–", ha="center", va="center", fontsize=8, color="#9CA3AF")
                    continue
                rgba = cmap(norm(value))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                color = "white" if luminance < 0.52 else DARK
                text = f"{value:.1f}" if title.endswith("MdAPE (%)") else f"{value:+.1f}".replace("-", "−")
                ax.text(j, i, text, ha="center", va="center", fontsize=7.2, color=color)
        return im

    norm_md = plt.Normalize(0, 20)
    norm_sg = TwoSlopeNorm(vcenter=0, vmin=-18, vmax=6)
    im00 = panel(
        axs[0, 0], md_off, plt.get_cmap("Blues"), norm_md,
        "Off-market · MdAPE (%)", True, False,
    )
    im01 = panel(
        axs[0, 1], md_list, plt.get_cmap("Blues"), norm_md,
        "List-aware · MdAPE (%)", False, False,
    )
    im10 = panel(
        axs[1, 0], sg_off, plt.get_cmap("RdBu_r"), norm_sg,
        "Off-market · median signed error (%)", True, True,
    )
    im11 = panel(
        axs[1, 1], sg_list, plt.get_cmap("RdBu_r"), norm_sg,
        "List-aware · median signed error (%)", False, True,
    )

    cb0 = fig.colorbar(im00, cax=caxes[0])
    cb0.set_label("MdAPE (%)", fontsize=8.5)
    cb0.ax.tick_params(labelsize=7.5)
    cb0.outline.set_visible(False)
    cb1 = fig.colorbar(im10, cax=caxes[1])
    cb1.set_label("Median signed error (%)", fontsize=8.5)
    cb1.ax.tick_params(labelsize=7.5)
    cb1.outline.set_visible(False)

    fig.suptitle(
        "Error by house type and price decile",
        x=0.12, y=0.965, ha="left", fontsize=15, fontweight="bold", color=DARK,
    )
    fig.text(
        0.12, 0.025,
        "Cells with fewer than 20 sales are greyed out. Columns are global sold-price deciles; each decile has about 605 homes.",
        fontsize=8.3, color=MUTED, ha="left",
    )
    fig.subplots_adjust(left=0.12, right=0.93, top=0.88, bottom=0.13)
    fig.savefig(OUT_DIR / "fulltest_type_price_heatmap.png", dpi=300, facecolor="white")
    fig.savefig(OUT_DIR / "fulltest_type_price_heatmap.pdf", facecolor="white")
    plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False, max_rows=8))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    valid, target, codes, edges, n_deciles, preds = load_predictions()
    frame = pd.read_csv(OUT_DIR / "fulltest_error_analysis_predictions.csv")

    draw_calibration(target, codes, n_deciles, preds)
    draw_cdf_from_frame(frame)
    draw_heatmap(frame, codes, n_deciles)
    print(f"wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
