"""Signed percentage error by listing month for the two fulltest checkpoints.

Scores the models trained at the 2026-02-01 fulltest cutoff on the matching
four-month validation window (Feb-May 2026, ``date_listed`` cohort) and plots
signed percentage error by the month each home was listed.

Signed error = (prediction - sold_price) / sold_price x 100; positive means the
model over-predicts.

Outputs (under reports/):
    fulltest_signed_error_by_month.png
    fulltest_signed_error_by_month.pdf
    fulltest_signed_error_by_month.csv
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
    {"id": "OffMarketRefinedV6", "label": "Off-market", "color": "#4C72B0"},
    {"id": "ListAwareTextStackV6Tuned", "label": "List-aware", "color": "#DD8452"},
)

MONTH_LABELS = {2: "Feb", 3: "Mar", 4: "Apr", 5: "May"}


def load_checkpoint(model_dir, model_id):
    matches = []
    for metadata_path in sorted(model_dir.glob(f"{model_id}_*.json")):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("training_cutoff") != CUTOFF.date().isoformat():
            continue
        model_path = metadata_path.with_suffix(".pkl")
        if model_path.exists():
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
    listings = data.load("listings")
    valid = listings[
        (listings["date_listed"] >= CUTOFF)
        & (listings["date_sold"] >= CUTOFF)
        & listings["date_sold"].notna()
    ].copy()
    if valid.empty:
        raise ValueError("No validation rows for the four-month fulltest window")
    return valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    valid = validation_frame()
    target = valid["sold_price"].to_numpy(float)
    period = valid["date_listed"].dt.to_period("M")
    months = sorted(period.unique())
    mask = {month: (period == month).to_numpy() for month in months}

    results = {}
    rows = []
    for spec in MODELS:
        model, metadata, model_path = load_checkpoint(args.model_dir, spec["id"])
        pred = np.asarray(
            model.predict(data.hide_answers(valid, model.TRACK))
        ).round().astype(float)
        signed = 100 * (pred - target) / target
        results[spec["id"]] = signed
        print(f"{spec['id']}: {len(signed):,} rows, median "
              f"{np.median(signed):+.2f}% (checkpoint {model_path.name})")

        for month in months:
            values = signed[mask[month]]
            low, high = harness.bootstrap_ci(values, np.median, B=5000, seed=42)
            rows.append({
                "model_id": spec["id"],
                "month": str(month),
                "N": int(values.size),
                "median_signed_pct": float(np.median(values)),
                "ci_low": low,
                "ci_high": high,
                "q1": float(np.percentile(values, 25)),
                "q3": float(np.percentile(values, 75)),
                "mean_signed_pct": float(np.mean(values)),
                "share_underpredicted": float(np.mean(values < 0)),
            })
    pd.DataFrame(rows).to_csv(
        args.out_dir / "fulltest_signed_error_by_month.csv", index=False
    )
    print(pd.DataFrame(rows)[
        ["model_id", "month", "N", "median_signed_pct", "ci_low", "ci_high"]
    ].to_string(index=False, float_format=lambda value: f"{value:+.2f}"))

    draw(results, months, mask, args.out_dir / "fulltest_signed_error_by_month.png")


def draw(results, months, mask, path):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    colors = {spec["id"]: spec["color"] for spec in MODELS}
    x = np.arange(len(months))
    positions = {"off": x - 0.19, "list": x + 0.19}

    fig, (ax_box, ax_med) = plt.subplots(
        1, 2, figsize=(13.0, 5.4), gridspec_kw={"width_ratios": [1.35, 1]}
    )

    extreme = 0.0
    for spec in MODELS:
        prefix = "off" if spec["id"] == "OffMarketRefinedV6" else "list"
        data_per_month = [results[spec["id"]][mask[month]] for month in months]
        for values in data_per_month:
            q1, q3 = np.percentile(values, [25, 75])
            fence = 1.5 * (q3 - q1)
            extreme = max(extreme,
                          abs(values[values >= q1 - fence].min()),
                          abs(values[values <= q3 + fence].max()))
        ax_box.boxplot(
            data_per_month,
            positions=positions[prefix],
            widths=0.36,
            patch_artist=True,
            manage_ticks=False,
            showfliers=False,
            boxprops=dict(facecolor=colors[spec["id"]],
                          edgecolor=colors[spec["id"]], alpha=0.55,
                          linewidth=1.0),
            medianprops=dict(color="#1a1a1a", linewidth=1.2),
            whiskerprops=dict(color=colors[spec["id"]], linewidth=0.9),
            capprops=dict(color=colors[spec["id"]], linewidth=0.9),
        )

    limit = extreme * 1.06
    ax_box.set_ylim(-limit, limit)
    ax_box.axhline(0, color="#333333", linewidth=1.0,
                   linestyle=(0, (4, 2)), zorder=0)
    ax_box.set_xticks(x)
    ax_box.set_xticklabels(
        [MONTH_LABELS[month.month] for month in months], fontsize=9.5
    )
    ax_box.set_xlim(-0.6, len(months) - 0.4)
    ax_box.set_xlabel("Listing month (2026)")
    ax_box.set_ylabel("Signed percentage error (%)")
    ax_box.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax_box.set_axisbelow(True)

    med_lim = 0.0
    for spec in MODELS:
        medians, lows, highs = [], [], []
        for month in months:
            values = results[spec["id"]][mask[month]]
            medians.append(float(np.median(values)))
            low, high = harness.bootstrap_ci(values, np.median, B=5000, seed=42)
            lows.append(low)
            highs.append(high)
        med_lim = max(med_lim, max(abs(value) for value in lows + highs))
        ax_med.errorbar(
            x, medians,
            yerr=[np.array(medians) - lows, np.array(highs) - np.array(medians)],
            fmt="o-", color=colors[spec["id"]], linewidth=1.5,
            markersize=5.2, elinewidth=1.0, capsize=2.8,
        )

    ax_med.set_xlim(-0.5, len(months) - 0.5)
    ax_med.set_ylim(-med_lim * 1.25, med_lim * 1.25)
    ax_med.axhline(0, color="#333333", linewidth=1.0,
                   linestyle=(0, (4, 2)), zorder=0)
    ax_med.set_xticks(x)
    ax_med.set_xticklabels(
        [MONTH_LABELS[month.month] for month in months], fontsize=9.5
    )
    ax_med.set_xlabel("Listing month (2026)")
    ax_med.set_ylabel("Median signed error (%)")
    ax_med.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax_med.set_axisbelow(True)

    handles = [
        Patch(facecolor=colors[spec["id"]], edgecolor=colors[spec["id"]],
              alpha=0.7, label=spec["label"])
        for spec in MODELS
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=2, frameon=False,
        fontsize=11, bbox_to_anchor=(0.5, 0.985), handlelength=1.7,
        handleheight=1.2, columnspacing=2.6, borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
