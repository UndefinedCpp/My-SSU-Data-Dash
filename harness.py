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

CLAIMED_BASELINE_OFF_MARKET = 20.8  # Reproduced value: 20.8428, CI: [20.398, 21.356]
CLAIMED_BASELINE_LIST_AWARE = 7.3  # Reproduced value: 7.2593, CI: [7.120, 7.414]

def predict_and_write(
    model: BaseModel,
    output_file_path: Path,
    verbose: bool = True
):
    track = model.TRACK
    my_print = logger.info if verbose else lambda *args, **kwargs: None
    my_print(f'Predicting with model "{model.NAME}" on track "{track}"')
    time_begin = time.perf_counter()

    my_print(f'Loading data...')
    homes = data.load(track)
    listings = data.load('listings')

    my_print('Training model on all listing data...')
    model.train(listings)  # Train with all available listing data

    my_print('Predicting...')
    predictions = model.predict(homes)

    pd.DataFrame({
        'ml_num': homes['ml_num'],
        'predicted_price': np.round(predictions)
    }).to_csv(output_file_path, index=False)

    time_elapsed = time.perf_counter() - time_begin
    if verbose:
        logger.success(f'Predictions written to {output_file_path} in {time_elapsed:.2f}s')


def bootstrap_ci(values: np.ndarray, f: Callable, B: int = 5000, alpha: float = 0.05, seed=42):
    rng = np.random.default_rng(seed)
    results = [
        f(rng.choice(values, size=len(values), replace=True))
        for _ in range(B)
    ]
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
        'N': len(pred_),
        'MAPE': MAPE,
        'MAPE_95CI': MAPE_95CI,
        'MdAPE': MdAPE,
        'MdAPE_95CI': MdAPE_95CI,
        'median_signed_err': median_signed_err,
        'mean_signed_err': mean_signed_err
    }


def test_significance(y: pd.Series, pred1: pd.Series, pred2: pd.Series, B=5000, alpha=0.05, seed=None):
    assert len(y) == len(pred1) == len(pred2), "Lengths must match"
    y   = y.to_numpy()
    p1  = pred1.to_numpy()
    p2  = pred2.to_numpy()
    n   = len(y)
    rng = np.random.default_rng(seed)

    diffs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        diffs[b] = np.median(y[idx] - p1[idx]) - np.median(y[idx] - p2[idx])

    lo, hi = np.percentile(diffs, [100*alpha/2, 100*(1-alpha/2)])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    p = min(p, 1.0)

    return {
        "delta": float(diffs.mean()),
        "ci": [float(lo), float(hi)],
        "p": float(p),
        "significant": not (lo <= 0 <= hi),          # 95% CI excludes 0
    }


def backtest(model: BaseModel, date_cutoff: pd.Period, verbose: bool = True, save_results: bool = False) -> str | pd.DataFrame:
    my_print = logger.info if verbose else (lambda *args, **kwargs: None)
    track = model.TRACK

    my_print(f'Preparing backtest data, cutoff: {date_cutoff}, track: "{track}"')
    listings = data.load('listings')
    train_data = listings[listings['date_sold'] < date_cutoff.start_time]
    valid_data = listings[listings['date_sold'].dt.to_period('M') == date_cutoff]

    my_print(f'Training started, dataset size: {len(train_data):,}')
    model.train(train_data)

    my_print(f'Validating, dataset size: {len(valid_data):,}')
    predictions = model.predict(data.hide_answers(valid_data, track))
    predictions = predictions.round()
    targets = valid_data['sold_price']

    experiment_df = pd.DataFrame({
        'pred': np.array(predictions),
        'target': np.array(targets)
    })

    metrics = report_metrics(predictions, targets)
    my_print(f'Finished backtest, MdAPE={metrics["MdAPE"]:.4f}')

    if save_results:
        date_string = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_directory = f'backtest/{model.get_id()}_{str(date_cutoff)}_{date_string}'
        os.makedirs(save_directory, exist_ok=False)
        # Save predictions
        experiment_df.to_csv(
            os.path.join(save_directory, 'predictions.csv'),
            index=False
        )
        # Save metrics
        with open(os.path.join(save_directory, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        # Save model file
        with open(os.path.join(save_directory, 'model.pkl'), 'wb') as f:
            pickle.dump(model, f)
        # Save python implementation (source code)
        with open(os.path.join(save_directory, 'model_source.py'), 'w') as f:
            try:
                f.write(inspect.getsource(model.__class__))
            except Exception as e:
                f.write(f"# Failed to get source code: {e}")
        my_print(f'Test results saved to: {save_directory}')

    if verbose:
        logger.success(f'Backtest completed for model: {model.get_id()}, cutoff: {date_cutoff}, track: "{track}"; metrics: MdAPE={metrics["MdAPE"]:.4f} (CI: [{metrics["MdAPE_95CI"][0]:.3f}, {metrics["MdAPE_95CI"][1]:.3f}])')

    if save_results:
        return save_directory
    else:
        return experiment_df
