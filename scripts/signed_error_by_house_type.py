"""Signed percentage error by house type for the two fulltest checkpoints.

Scores the models trained at the 2026-02-01 fulltest cutoff on the matching
four-month validation window (Feb-May 2026, ``date_listed`` cohort) and draws
grouped boxplots of signed percentage error by house type.

Signed error = (prediction - sold_price) / sold_price x 100; positive means the
model over-predicts.

Outputs (under reports/):
    fulltest_signed_error_by_house_type.png
    fulltest_signed_error_by_house_type.pdf
    fulltest_signed_error_by_house_type.csv
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

TWO_LINE = {
    "Condo Townhouse": "Condo\nTownhouse",
    "Freehold Townhouse": "Freehold\nTownhouse",
}


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
    house_type = valid["house_type_name"].fillna("Unknown").astype(str)
    order = list(house_type.value_counts().index)
    mask = {name: (house_type == name).to_numpy() for name in order}

    results = {}
    rows = []
    for spec in MODELS:
        model, metadata, model_path = load_checkpoint(args.model_dir, spec["id"])
        hidden = data.hide_answers(valid, model.TRACK)
        pred = np.asarray(model.predict(hidden)).round().astype(float)
        signed = 100 * (pred - target) / target
        results[spec["id"]] = signed
        print(f"{spec['id']}: {len(signed):,} rows, median "
              f"{np.median(signed):+.2f}% (checkpoint {model_path.name})")

        for name in order:
            values = signed[mask[name]]
            low, high = harness.bootstrap_ci(values, np.median, B=5000, seed=42)
            rows.append({
                "model_id": spec["id"],
                "house_type": name,
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
        args.out_dir / "fulltest_signed_error_by_house_type.csv", index=False
    )
    print(pd.DataFrame(rows)[
        ["model_id", "house_type", "N", "median_signed_pct", "ci_low", "ci_high"]
    ].to_string(index=False, float_format=lambda value: f"{value:+.2f}"))

    draw(results, mask, order, args.out_dir / "fulltest_signed_error_by_house_type.png")


def draw(results, mask, order, path):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    colors = {spec["id"]: spec["color"] for spec in MODELS}
    x = np.arange(len(order))
    positions = {"off": x - 0.19, "list": x + 0.19}
    width = 0.36

    fig, ax = plt.subplots(figsize=(12.6, 5.4))
    extreme = 0.0
    for spec in MODELS:
        prefix = "off" if spec["id"] == "OffMarketRefinedV6" else "list"
        data_per_type = [results[spec["id"]][mask[name]] for name in order]
        for values in data_per_type:
            q1, q3 = np.percentile(values, [25, 75])
            fence = 1.5 * (q3 - q1)
            extreme = max(extreme, abs(values[values >= q1 - fence].min()),
                          abs(values[values <= q3 + fence].max()))
        ax.boxplot(
            data_per_type,
            positions=positions[prefix],
            widths=width,
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
    ax.set_ylim(-limit, limit)
    ax.axhline(0, color="#333333", linewidth=1.0,
               linestyle=(0, (4, 2)), zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([TWO_LINE.get(name, name) for name in order],
                       fontsize=9.0, linespacing=1.25)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel("Signed percentage error (%)")
    ax.set_xlabel("House type")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

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
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
