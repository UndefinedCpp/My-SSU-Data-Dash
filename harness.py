"""
Module for managing and running models
"""

from typing import Callable

import pandas as pd
import numpy as np
from models.BaseModel import BaseModel, Track
from model_hub import gather_models, load_model
from pathlib import Path
import data
from loguru import logger
import time
import json
from datetime import datetime
import os
import pickle
import inspect
from tqdm import tqdm
import model_hub
from scipy.stats import wilcoxon

CLAIMED_BASELINE_OFF_MARKET = 20.8  # Reproduced value: 20.8428, CI: [20.398, 21.356]
CLAIMED_BASELINE_LIST_AWARE = 7.3  # Reproduced value: 7.2593, CI: [7.120, 7.414]


def predict_and_write(model: BaseModel, output_file_path: Path, verbose: bool = True):
    track = model.TRACK
    my_print = logger.info if verbose else lambda *args, **kwargs: None
    my_print(f'Predicting with model "{model.NAME}" on track "{track}"')
    time_begin = time.perf_counter()

    my_print(f"Loading data...")
    homes = data.load(track)
    listings = data.load("listings")

    my_print("Training model on all listing data...")
    model.train(listings)  # Train with all available listing data

    my_print("Predicting...")
    predictions = model.predict(homes)

    pd.DataFrame(
        {"ml_num": homes["ml_num"], "predicted_price": np.round(predictions)}
    ).to_csv(output_file_path, index=False)

    time_elapsed = time.perf_counter() - time_begin
    if verbose:
        logger.success(
            f"Predictions written to {output_file_path} in {time_elapsed:.2f}s"
        )


def bootstrap_ci(
    values: np.ndarray, f: Callable, B: int = 5000, alpha: float = 0.05, seed=42
):
    rng = np.random.default_rng(seed)
    results = [f(rng.choice(values, size=len(values), replace=True)) for _ in range(B)]
    lower_ci = np.percentile(results, 100 * alpha / 2)
    upper_ci = np.percentile(results, 100 * (1 - alpha / 2))
    return float(lower_ci), float(upper_ci)


def report_metrics(predictions: pd.Series, target: pd.Series):
    pred_ = np.array(predictions.values)
    target_ = np.array(target.values)
    signed_err = 100 * (pred_ - target_) / target_  # percentage error
    abs_err = np.abs(signed_err)

    MAPE = np.mean(abs_err)
    MAPE_95CI = bootstrap_ci(abs_err, np.mean)
    MdAPE = np.median(abs_err)
    MdAPE_95CI = bootstrap_ci(abs_err, np.median)
    median_signed_err = np.median(signed_err)
    mean_signed_err = np.mean(signed_err)

    return {
        "N": len(pred_),
        "MAPE": MAPE,
        "MAPE_95CI": MAPE_95CI,
        "MdAPE": MdAPE,
        "MdAPE_95CI": MdAPE_95CI,
        "median_signed_err": median_signed_err,
        "mean_signed_err": mean_signed_err,
    }


def test_significance(
    y: pd.Series, pred1: pd.Series, pred2: pd.Series, B=5000, alpha=0.05, seed=None
):
    assert len(y) == len(pred1) == len(pred2), "Lengths must match"
    y = y.to_numpy()
    p1 = pred1.to_numpy()
    p2 = pred2.to_numpy()
    n = len(y)
    rng = np.random.default_rng(seed)

    diffs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        diffs[b] = np.median(y[idx] - p1[idx]) - np.median(y[idx] - p2[idx])

    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    p = min(p, 1.0)

    return {
        "delta": float(diffs.mean()),
        "ci": [float(lo), float(hi)],
        "p": float(p),
        "significant": not (lo <= 0 <= hi),  # 95% CI excludes 0
    }


def backtest(
    model: BaseModel,
    date_cutoff: pd.Period,
    verbose: bool = True,
    save_results: bool = False,
) -> str | pd.DataFrame:
    my_print = logger.info if verbose else (lambda *args, **kwargs: None)
    track = model.TRACK

    my_print(f'Preparing backtest data, cutoff: {date_cutoff}, track: "{track}"')
    listings = data.load("listings")
    train_data = listings[listings["date_sold"] < date_cutoff.start_time]
    valid_data = listings[listings["date_sold"].dt.to_period("M") == date_cutoff]

    my_print(f"Training started, dataset size: {len(train_data):,}")
    model.train(train_data)

    my_print(f"Validating, dataset size: {len(valid_data):,}")
    predictions = model.predict(data.hide_answers(valid_data, track))
    predictions = predictions.round()
    targets = valid_data["sold_price"]

    experiment_df = pd.DataFrame(
        {"pred": np.array(predictions), "target": np.array(targets)}
    )

    metrics = report_metrics(predictions, targets)
    my_print(f'Finished backtest, MdAPE={metrics["MdAPE"]:.4f}')

    if save_results:
        date_string = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_directory = f"backtest/{model.get_id()}_{str(date_cutoff)}_{date_string}"
        os.makedirs(save_directory, exist_ok=False)
        # Save predictions
        experiment_df.to_csv(
            os.path.join(save_directory, "predictions.csv"), index=False
        )
        # Save metrics
        with open(os.path.join(save_directory, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        # Save model file
        with open(os.path.join(save_directory, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
        # Save python implementation (source code)
        with open(os.path.join(save_directory, "model_source.py"), "w") as f:
            try:
                f.write(inspect.getsource(model.__class__))
            except Exception as e:
                f.write(f"# Failed to get source code: {e}")
        my_print(f"Test results saved to: {save_directory}")

    if verbose:
        logger.success(
            f'Backtest completed for model: {model.get_id()}, cutoff: {date_cutoff}, track: "{track}"; metrics: MdAPE={metrics["MdAPE"]:.4f} (CI: [{metrics["MdAPE_95CI"][0]:.3f}, {metrics["MdAPE_95CI"][1]:.3f}])'
        )

    if save_results:
        return save_directory
    else:
        return experiment_df


def benchmark(track):
    """
    Grab all models enabled for the specified track and run backtest for each
    of them. Finally generate a leading board and a benchmark report.
    """
    # Get all models enabled for the specified track
    enabled_models = [
        m
        for m in model_hub.gather_models(Path("models"))
        if m.TRACK == track and m.ENABLED
    ]
    logger.info(
        f"Gathered {len(enabled_models)} model(s): {[m.__name__ for m in enabled_models]}"
    )
    # Run backtest for each enabled model
    date_cutoffs = pd.period_range("2025-12", "2026-05", freq="M")
    progress_bar = tqdm(
        desc="Running benchmarks", total=len(enabled_models) * len(date_cutoffs)
    )

    pooled_metrics = {}
    prediction_results = {}

    for model_class in enabled_models:
        model = model_class()
        experiment_dataframes = []
        for dc in date_cutoffs:
            experiment_df = backtest(model, dc, verbose=False, save_results=False)
            experiment_dataframes.append(experiment_df)
            progress_bar.update(1)
        # Create pooled statistics
        pooled_df = pd.concat(experiment_dataframes, ignore_index=True)
        metrics = report_metrics(pooled_df["pred"], pooled_df["target"])
        # Store the pooled metrics and prediction results for the current model
        pooled_metrics[model.get_id()] = metrics
        prediction_results[model.get_id()] = pooled_df[["pred", "target"]]
    progress_bar.close()

    # Check if models have significant differences in their performance
    logger.info(f"Running Wilcoxon signed-rank tests for model comparisons")
    wilcoxon_test_results = {}
    model_ids = list(pooled_metrics.keys())

    for i, model_id_i in enumerate(model_ids):
        for j, model_id_j in enumerate(model_ids):
            if i >= j:
                continue

            preds_i = prediction_results[model_id_i]["pred"]
            preds_j = prediction_results[model_id_j]["pred"]
            targets = prediction_results[model_id_i]["target"]
            # Test significance with Wilcoxon signed-rank test.
            # Compute absolute percentage error
            ape_i = np.abs(preds_i - targets) / np.abs(targets) * 100
            ape_j = np.abs(preds_j - targets) / np.abs(targets) * 100
            diffs = ape_i - ape_j  # paired differences
            # two-sided: H0: median(diffs) == 0 vs H1: median(diffs) != 0
            stat, p = wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
            is_significant = bool(p < 0.05)
            # effect size & direction (this is what says "which is better")
            med_diff = np.median(diffs)
            better = model_id_j if med_diff > 0 else model_id_i
            wilcoxon_test_results[(model_id_i, model_id_j)] = {
                "delta": med_diff,
                "p": p,
                "significant": is_significant,
                "better": None if not is_significant else better,
            }

    # Save results
    date_string = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file_name = f"report_{track}_{date_string}.json"
    os.makedirs("backtest", exist_ok=True)
    report_path = os.path.join("backtest", report_file_name)
    with open(report_path, "w") as f:
        json.dump(
            {
                "metrics": pooled_metrics,
                "wilcoxon_test": {
                    f"{k[0]}:{k[1]}": v for k, v in wilcoxon_test_results.items()
                },
            },
            f,
        )
    logger.info(f"Metrics and Wilcoxon test report saved to: {report_path}")

    # Finally make a ranking of models based on the results
    # Count significant wins/losses per model
    wins = {m: 0 for m in model_ids}
    losses = {m: 0 for m in model_ids}
    for (a, b), r in wilcoxon_test_results.items():
        if not r["significant"]:
            continue
        winner = r["better"]
        loser = b if winner == a else a
        wins[winner] += 1
        losses[loser] += 1
    # Primary ranking key: MdAPE, lower is better
    leaderboard_print_lst = ["Leaderboard:"]
    leaderboard_print_lst.append(
        f'{"#":>2} {"Tag":<10} {"Model":<30} {"MdAPE":>7} {"MAPE":>7} {"Md Err":>7}'
    )
    leaderboard_print_lst.append("-" * 70)
    leaderboard = sorted(
        model_ids, key=lambda r: (pooled_metrics[r]["MdAPE"], -wins[r])
    )
    rank_counter = 1
    for i, mid in enumerate(leaderboard):
        MdAPE = pooled_metrics[mid]["MdAPE"]
        MAPE = pooled_metrics[mid]["MAPE"]
        display_name = f"{mid[:27]}..." if len(mid) > 30 else mid
        median_signed_err = pooled_metrics[mid]["median_signed_err"]
        tag = model_hub.load_model(mid).MODEL_TAG[:10]
        leaderboard_print_lst.append(
            f"{rank_counter:>2} {tag:<10} {display_name:<30} {MdAPE:>7.3f} {MAPE:>7.3f} {median_signed_err:>+7.3f}"
        )
        # Increase rank counter if this is not a tie with the previous model
        if i < len(leaderboard) - 1:
            next_mid = leaderboard[i + 1]
            is_significantly_better = wilcoxon_test_results.get(
                (mid, next_mid), {}
            ).get("significant", False) or wilcoxon_test_results.get(
                (next_mid, mid), {}
            ).get(
                "significant", False
            )
            if is_significantly_better:
                rank_counter += 1
    logger.info("\n".join(leaderboard_print_lst))
