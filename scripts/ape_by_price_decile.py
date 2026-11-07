"""Evaluate the two saved fulltest checkpoints by sold-price decile.

Loads the checkpoints trained at the fulltest cutoff (2026-02-01), scores them
on the matching four-month validation window (Feb-May 2026, ``date_listed``
cohort), and draws absolute percentage error (APE) by sold-price decile.

Outputs (under reports/):
    fulltest_ape_by_price_decile.png
    fulltest_ape_by_price_decile.csv
    fulltest_ape_by_price_decile.md
    fulltest_target_model_metrics.csv
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import data  # noqa: E402
import harness  # noqa: E402

CUTOFF = pd.Timestamp("2026-02-01")
DEFAULT_MODEL_DIR = REPO / "saved_models/fulltest"
DEFAULT_OUT_DIR = REPO / "reports"

MODELS = (
    {
        "id": "OffMarketRefinedV6",
        "label": "Off-market",
        "color": "#4C72B0",
    },
    {
        "id": "ListAwareTextStackV6Tuned",
        "label": "List-aware",
        "color": "#DD8452",
    },
)


def load_checkpoint(model_dir, model_id):
    """Return the newest checkpoint for ``model_id`` fitted at the fulltest cutoff."""
    matches = []
    for metadata_path in sorted(model_dir.glob(f"{model_id}_*.json")):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("training_cutoff") != CUTOFF.date().isoformat():
            continue
        model_path = metadata_path.with_suffix(".pkl")
        if not model_path.exists():
            continue
        matches.append((metadata_path, model_path, metadata))
    if not matches:
        raise FileNotFoundError(
            f"No {model_id} checkpoint with training_cutoff="
            f"{CUTOFF.date().isoformat()} in {model_dir}"
        )
    metadata_path, model_path, metadata = matches[-1]
    with open(model_path, "rb") as handle:
        model = pickle.load(handle)
    return model, metadata, model_path


def validation_frame():
    """Exactly the harness.fulltest validation rows for the default cohort."""
    listings = data.load("listings")
    valid = listings[
        (listings["date_listed"] >= CUTOFF)
        & (listings["date_sold"] >= CUTOFF)
        & listings["date_sold"].notna()
    ].copy()
    if valid.empty:
        raise ValueError("No validation rows for the four-month fulltest window")
    return valid


def predict(model, valid):
    """Predict and round, matching the harness evaluation protocol."""
    hidden = data.hide_answers(valid, model.TRACK)
    return pd.Series(np.asarray(model.predict(hidden))).round().to_numpy(float)


def compact_price(value):
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    return f"${value / 1_000:.0f}k"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    valid = validation_frame()
    target = valid["sold_price"].to_numpy(float)
    print(f"Validation rows: {len(valid):,} (listed and sold on/after "
          f"{CUTOFF.date().isoformat()})")

    codes, bin_edges = pd.qcut(
        target, 10, labels=False, retbins=True, duplicates="drop"
    )
    n_deciles = len(bin_edges) - 1
    decile_names = [f"D{i + 1}" for i in range(n_deciles)]

    results = {}
    overall_rows = []
    for spec in MODELS:
        model, metadata, model_path = load_checkpoint(args.model_dir, spec["id"])
        if model.TRACK != metadata["track"]:
            raise ValueError(f"{spec['id']}: checkpoint track mismatch")
        pred = predict(model, valid)
        ape = 100 * np.abs(pred - target) / target
        signed = 100 * (pred - target) / target
        metrics = harness.report_metrics(pd.Series(pred), pd.Series(target))
        recorded = metadata.get("metrics") or {}
        for key in ("N", "MdAPE", "MAPE"):
            if key in recorded and not np.isclose(metrics[key], recorded[key], rtol=0, atol=1e-9):
                raise AssertionError(
                    f"{spec['id']}: reloaded {key}={metrics[key]} != recorded "
                    f"{recorded[key]}; checkpoint does not reproduce its metadata"
                )
        results[spec["id"]] = {
            "spec": spec,
            "metrics": metrics,
            "metadata": metadata,
            "model_path": model_path,
            "ape": ape,
            "signed": signed,
        }
        overall_rows.append({
            "model_id": spec["id"],
            "track": metadata["track"],
            "training_cutoff": metadata["training_cutoff"],
            "N": metrics["N"],
            "MdAPE": metrics["MdAPE"],
            "MdAPE_95CI_low": metrics["MdAPE_95CI"][0],
            "MdAPE_95CI_high": metrics["MdAPE_95CI"][1],
            "MAPE": metrics["MAPE"],
            "median_signed_err": metrics["median_signed_err"],
            "mean_signed_err": metrics["mean_signed_err"],
        })
        print(f"{spec['id']}: {metrics['N']:,} rows, MdAPE={metrics['MdAPE']:.4f}, "
              f"MAPE={metrics['MAPE']:.4f} (checkpoint {model_path.name})")

    pd.DataFrame(overall_rows).to_csv(
        args.out_dir / "fulltest_target_model_metrics.csv", index=False
    )

    # Per-decile summaries with bootstrap CIs on the median APE.
    decile_rows = []
    decile_stats = {
        spec["id"]: {"med": [], "lo": [], "hi": [], "mean": []} for spec in MODELS
    }
    for index in range(n_deciles):
        mask = codes == index
        median_price = float(np.median(target[mask]))
        row = {
            "price_decile": decile_names[index],
            "price_low": float(bin_edges[index]),
            "price_high": float(bin_edges[index + 1]),
            "median_price": median_price,
            "N": int(mask.sum()),
        }
        for spec in MODELS:
            ape = results[spec["id"]]["ape"]
            signed = results[spec["id"]]["signed"]
            median_ape = float(np.median(ape[mask]))
            low, high = harness.bootstrap_ci(ape[mask], np.median, B=5000, seed=42)
            decile_stats[spec["id"]]["med"].append(median_ape)
            decile_stats[spec["id"]]["lo"].append(low)
            decile_stats[spec["id"]]["hi"].append(high)
            decile_stats[spec["id"]]["mean"].append(float(np.mean(ape[mask])))
            prefix = "off" if spec["id"] == "OffMarketRefinedV6" else "list"
            row[f"{prefix}_MdAPE"] = median_ape
            row[f"{prefix}_MdAPE_95CI_low"] = low
            row[f"{prefix}_MdAPE_95CI_high"] = high
            row[f"{prefix}_MAPE"] = float(np.mean(ape[mask]))
            row[f"{prefix}_median_signed_err"] = float(np.median(signed[mask]))
        decile_rows.append(row)
    decile_df = pd.DataFrame(decile_rows)
    decile_df.to_csv(args.out_dir / "fulltest_ape_by_price_decile.csv", index=False)

    draw_figure_summary(
        decile_df, decile_stats, results, codes, n_deciles,
        args.out_dir / "fulltest_ape_by_price_decile.png",
    )
    draw_figure_full(
        decile_df, decile_stats, results, codes, n_deciles,
        args.out_dir / "fulltest_ape_by_price_decile_full.png",
    )
    write_summary(decile_df, overall_rows,
                  args.out_dir / "fulltest_ape_by_price_decile.md")
    print(f"\nWrote figure, CSV and summary to {args.out_dir}")
    print(decile_df[[
        "price_decile", "N", "off_MdAPE", "list_MdAPE",
        "off_MdAPE_95CI_low", "off_MdAPE_95CI_high",
        "list_MdAPE_95CI_low", "list_MdAPE_95CI_high",
    ]].to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def _add_median_panel(ax, decile_df, decile_stats, colors, n_deciles):
    x = np.arange(n_deciles)
    for spec in MODELS:
        stats = decile_stats[spec["id"]]
        med = np.asarray(stats["med"])
        low = np.asarray(stats["lo"])
        high = np.asarray(stats["hi"])
        ax.errorbar(
            x, med, yerr=[med - low, high - med], fmt="o-",
            color=colors[spec["id"]], linewidth=1.5, markersize=4.8,
            elinewidth=1.0, capsize=2.6,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(decile_df["price_decile"], fontsize=8.2)
    ax.set_xlim(-0.5, n_deciles - 0.5)
    upper = max(max(stats["hi"]) for stats in decile_stats.values())
    ax.set_ylim(0, upper * 1.16)
    ax.set_ylabel("Median APE (%)")
    ax.set_xlabel("Sold-price decile")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def _add_legend(fig, colors):
    handles = [
        Patch(facecolor=colors[spec["id"]], edgecolor=colors[spec["id"]],
              alpha=0.7, label=spec["label"])
        for spec in MODELS
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=2, frameon=False,
        fontsize=11, bbox_to_anchor=(0.5, 0.99), handlelength=1.7,
        handleheight=1.2, columnspacing=2.6, borderaxespad=0.0,
    )


def draw_figure_summary(decile_df, decile_stats, results, codes, n_deciles, path):
    """Headline figure: boxes with 1.5xIQR whiskers, outliers not drawn."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    colors = {spec["id"]: spec["color"] for spec in MODELS}
    x = np.arange(n_deciles)

    fig, (ax_box, ax_med) = plt.subplots(
        1, 2, figsize=(13.0, 5.4), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    positions = {"off": x - 0.19, "list": x + 0.19}
    whisker_tops = []
    for spec in MODELS:
        prefix = "off" if spec["id"] == "OffMarketRefinedV6" else "list"
        data_per_decile = [
            results[spec["id"]]["ape"][codes == index] for index in range(n_deciles)
        ]
        for values in data_per_decile:
            q1, q3 = np.percentile(values, [25, 75])
            fence = q3 + 1.5 * (q3 - q1)
            whisker_tops.append(float(values[values <= fence].max()))
        ax_box.boxplot(
            data_per_decile,
            positions=positions[prefix],
            widths=0.36,
            patch_artist=True,
            manage_ticks=False,
            showfliers=False,
            boxprops=dict(facecolor=colors[spec["id"]], edgecolor=colors[spec["id"]],
                          alpha=0.55, linewidth=1.0),
            medianprops=dict(color="#1a1a1a", linewidth=1.2),
            whiskerprops=dict(color=colors[spec["id"]], linewidth=0.9),
            capprops=dict(color=colors[spec["id"]], linewidth=0.9),
        )
    ax_box.set_ylim(0, max(whisker_tops) * 1.04)
    ax_box.set_xticks(x)
    ax_box.set_xticklabels(
        [f"{row.price_decile}\n{compact_price(row.median_price)}"
         for row in decile_df.itertuples()],
        fontsize=8.0, linespacing=1.3,
    )
    ax_box.set_ylabel("Absolute percentage error (%)")
    ax_box.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax_box.set_axisbelow(True)

    _add_median_panel(ax_med, decile_df, decile_stats, colors, n_deciles)
    _add_legend(fig, colors)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def draw_figure_full(decile_df, decile_stats, results, codes, n_deciles, path):
    """Supplementary figure: every outlier drawn, extreme tail on a broken axis."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    colors = {spec["id"]: spec["color"] for spec in MODELS}
    x = np.arange(n_deciles)
    break_at = 50.0

    fig = plt.figure(figsize=(13.0, 5.8))
    grid = fig.add_gridspec(
        2, 2, height_ratios=[1.55, 5.0], width_ratios=[1.35, 1],
        hspace=0.06, wspace=0.18,
    )
    ax_top = fig.add_subplot(grid[0, 0])
    ax_box = fig.add_subplot(grid[1, 0], sharex=ax_top)
    ax_med = fig.add_subplot(grid[:, 1])

    positions = {"off": x - 0.19, "list": x + 0.19}
    whisker_tops = []
    flier_max = break_at
    for spec in MODELS:
        prefix = "off" if spec["id"] == "OffMarketRefinedV6" else "list"
        data_per_decile = [
            results[spec["id"]]["ape"][codes == index] for index in range(n_deciles)
        ]
        for values in data_per_decile:
            q1, q3 = np.percentile(values, [25, 75])
            fence = q3 + 1.5 * (q3 - q1)
            whisker_tops.append(float(values[values <= fence].max()))
            flier_max = max(flier_max, float(values.max()))
        ax_box.boxplot(
            data_per_decile,
            positions=positions[prefix],
            widths=0.36,
            patch_artist=True,
            manage_ticks=False,
            showfliers=True,
            boxprops=dict(facecolor=colors[spec["id"]], edgecolor=colors[spec["id"]],
                          alpha=0.55, linewidth=1.0),
            medianprops=dict(color="#1a1a1a", linewidth=1.2),
            whiskerprops=dict(color=colors[spec["id"]], linewidth=0.9),
            capprops=dict(color=colors[spec["id"]], linewidth=0.9),
            flierprops=dict(marker="o", markersize=2.6,
                            markerfacecolor=colors[spec["id"]],
                            markeredgecolor="none", alpha=0.6),
        )
        for index, values in enumerate(data_per_decile):
            highs = values[values > break_at]
            if highs.size:
                jitter = (np.linspace(-0.05, 0.05, highs.size)
                          if highs.size > 1 else np.zeros(1))
                ax_top.scatter(
                    positions[prefix][index] + jitter, highs, s=14,
                    color=colors[spec["id"]], alpha=0.7, edgecolors="none",
                )

    ax_box.set_ylim(0, break_at)
    ax_box.set_yticks([0, 10, 20, 30, 40])
    ax_box.set_xticks(x)
    ax_box.set_xticklabels(
        [f"{row.price_decile}\n{compact_price(row.median_price)}"
         for row in decile_df.itertuples()],
        fontsize=8.0, linespacing=1.3,
    )
    ax_box.set_ylabel("Absolute percentage error (%)")
    ax_box.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax_box.set_axisbelow(True)

    ax_top.set_ylim(break_at, flier_max * 1.03)
    ax_top.set_yticks([70, 110, 150])
    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_top.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax_top.set_axisbelow(True)
    ax_top.spines["bottom"].set_visible(False)
    ax_box.spines["top"].set_visible(False)
    size = 0.02
    for axis, y in ((ax_top, 0.0), (ax_box, 1.0)):
        axis.plot((-size, size), (y - size, y + size),
                  transform=axis.transAxes, color="black",
                  clip_on=False, linewidth=0.8)

    _add_median_panel(ax_med, decile_df, decile_stats, colors, n_deciles)
    _add_legend(fig, colors)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def write_summary(decile_df, overall_rows, path):
    overall = pd.DataFrame(overall_rows).set_index("model_id")
    off = overall.loc["OffMarketRefinedV6"]
    lst = overall.loc["ListAwareTextStackV6Tuned"]

    lines = [
        "# Absolute percentage error by sold-price decile: Feb–May 2026 fulltest",
        "",
        f"Validation set: {int(off['N']):,} sales listed and sold on/after "
        "2026-02-01 (`date_listed` cohort), scored with the checkpoints trained "
        "on sales before that cutoff. Predictions are rounded as in the harness.",
        "",
        "## Overall",
        "",
        "| Model | N | MdAPE | MdAPE 95% CI | MAPE | Median signed err |",
        "|---|---:|---:|---|---:|---:|",
        f"| OffMarketRefinedV6 | {int(off['N']):,} | {off['MdAPE']:.4f} | "
        f"[{off['MdAPE_95CI_low']:.3f}, {off['MdAPE_95CI_high']:.3f}] | "
        f"{off['MAPE']:.4f} | {off['median_signed_err']:+.3f} |",
        f"| ListAwareTextStackV6Tuned | {int(lst['N']):,} | {lst['MdAPE']:.4f} | "
        f"[{lst['MdAPE_95CI_low']:.3f}, {lst['MdAPE_95CI_high']:.3f}] | "
        f"{lst['MAPE']:.4f} | {lst['median_signed_err']:+.3f} |",
        "",
        "## By decile",
        "",
        "| Decile | Sold price range | Median price | N | Off MdAPE (CI) | List MdAPE (CI) | "
        "Off − List |",
        "|---|---|---:|---:|---|---|---:|",
    ]
    for row in decile_df.itertuples():
        lines.append(
            f"| {row.price_decile} | {compact_price(row.price_low)}–"
            f"{compact_price(row.price_high)} | {compact_price(row.median_price)} | "
            f"{row.N} | {row.off_MdAPE:.3f} "
            f"[{row.off_MdAPE_95CI_low:.3f}, {row.off_MdAPE_95CI_high:.3f}] | "
            f"{row.list_MdAPE:.3f} "
            f"[{row.list_MdAPE_95CI_low:.3f}, {row.list_MdAPE_95CI_high:.3f}] | "
            f"{row.off_MdAPE - row.list_MdAPE:+.3f} |"
        )
    first, last = decile_df.iloc[0], decile_df.iloc[-1]
    lines += [
        "",
        "## Reading",
        "",
        f"- Off-market MdAPE: {first.off_MdAPE:.2f}% in D1, a low of "
        f"{decile_df['off_MdAPE'].min():.2f}% at "
        f"{decile_df.loc[decile_df['off_MdAPE'].idxmin(), 'price_decile']}, then "
        f"{last.off_MdAPE:.2f}% in D10 "
        f"({last.off_MdAPE - first.off_MdAPE:+.2f} pp vs D1).",
        f"- List-aware MdAPE stays within "
        f"{decile_df['list_MdAPE'].min():.2f}–{decile_df['list_MdAPE'].max():.2f}% "
        f"across deciles ({first.list_MdAPE:.2f}% in D1, "
        f"{last.list_MdAPE:.2f}% in D10).",
        f"- The off-market penalty over list-aware grows from "
        f"{first.off_MdAPE - first.list_MdAPE:.2f} pp in D1 to "
        f"{last.off_MdAPE - last.list_MdAPE:.2f} pp in D10 "
        f"(largest: {(decile_df['off_MdAPE'] - decile_df['list_MdAPE']).max():.2f} pp in "
        f"{decile_df.loc[(decile_df['off_MdAPE'] - decile_df['list_MdAPE']).idxmax(), 'price_decile']}).",
        f"- Median signed error: off-market {first.off_median_signed_err:+.2f}% in D1 "
        f"vs {last.off_median_signed_err:+.2f}% in D10 (over-predicts cheap homes, "
        f"under-predicts expensive ones); list-aware "
        f"{first.list_median_signed_err:+.2f}% vs {last.list_median_signed_err:+.2f}%.",
        "",
        "Figure: `reports/fulltest_ape_by_price_decile.png` (and `.pdf`).",
        "Per-decile numbers: `reports/fulltest_ape_by_price_decile.csv`.",
        "Overall metrics: `reports/fulltest_target_model_metrics.csv`.",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
