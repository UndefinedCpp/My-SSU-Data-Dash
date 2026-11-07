"""Independently validate completed ablation artifacts and draw a summary figure.

Run from the project root:
python scripts/verify_offmarket_ablations.py reports/ablations/RUN_DIRECTORY

Restores native branch dtypes before reblending saved CSV values, and writes
lossless NPZ branch arrays plus a dtype manifest for future reproductions.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def summary(pred, target):
    signed = 100 * (pred - target) / target
    ape = np.abs(signed)
    return {'N': len(target), 'MdAPE': np.median(ape), 'MAPE': np.mean(ape),
            'P90_APE': np.quantile(ape, .9), 'median_signed_err': np.median(signed),
            'mean_signed_err': np.mean(signed)}


def reblend(b, c, s):
    incumbent = np.exp(.5 * (np.log(b) + np.log(c)))
    return np.round(np.exp(.75 * np.log(incumbent) + .25 * np.log(s)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    manifest = json.loads((folder / 'provenance.json').read_text())
    require(manifest['status'] == 'completed', 'Run is not completed')
    require(manifest['fresh_full_refit_matches_checkpoint'], 'No matched full-refit control')
    require(manifest['additional_learner_fits'] == 10, 'Unexpected fit count')
    for name, recorded in manifest['artifacts_sha256'].items():
        require(digest(folder / name) == recorded, f'Artifact hash mismatch: {name}')
    pred = pd.read_csv(folder / 'predictions.csv', float_precision='round_trip')
    components = pd.read_csv(folder / 'component_predictions.csv', float_precision='round_trip')
    overall = pd.read_csv(folder / 'overall_metrics.csv', float_precision='round_trip').set_index('variant')
    segments = pd.read_csv(folder / 'segment_metrics.csv', float_precision='round_trip')
    audits = json.loads((folder / 'feature_audit.json').read_text())
    timing = pd.read_csv(folder / 'timings.csv')
    require(len(pred) == 6047 and pred.ml_num.is_unique, 'Wrong validation IDs')
    require(pred.ml_num.equals(components.ml_num) and pred.sold_price.equals(components.sold_price), 'ID/target alignment mismatch')
    require(np.array_equal(pred.full, pred.full_refit), 'Fresh full refit differs from saved full baseline')
    require(len(timing[timing.variant != 'checkpoint']) == 10, 'Missing learner timings')
    target = pred.sold_price.to_numpy()
    for name, row in overall.iterrows():
        values = pred[name].to_numpy()
        require(np.isfinite(values).all() and (values > 0).all(), f'Invalid prediction: {name}')
        for key, value in summary(values, target).items():
            require(np.isclose(value, row[key], rtol=0, atol=1e-10), f'Metric mismatch: {name}/{key}')
        require(np.isclose(row.delta_MdAPE_pp, row.MdAPE - overall.loc['full', 'MdAPE'], atol=1e-12), 'Delta mismatch')
        require(row.paired_row_CI_low <= row.paired_row_CI_high, 'Invalid paired CI')
        require(row.paired_coordinate_cluster_CI_low <= row.paired_coordinate_cluster_CI_high, 'Invalid cluster CI')
    for _, row in segments.iterrows():
        mask = np.ones(len(pred), dtype=bool) if row.dimension == 'overall' else pred[row.dimension].eq(row.segment).to_numpy()
        require(mask.sum() == row.N and mask.any(), 'Segment count mismatch')
        for key, value in summary(pred[row.variant].to_numpy()[mask], target[mask]).items():
            require(np.isclose(value, row[key], rtol=0, atol=1e-10), f'Segment metric mismatch: {row.variant}/{row.segment}/{key}')
    for variant, prefix, count in [('full_refit', '', 0), ('no_comparables', 'comp_', 14), ('no_building', 'bldg_', 24)]:
        for branch in 'BCS':
            audit = audits[f'{variant}/{branch}']
            removed = audit['removed_learner_columns']
            require(len(removed) == count, f'Wrong removed feature count: {variant}/{branch}')
            require(all(c.startswith(prefix) for c in removed), 'Wrong feature prefix')
            require(audit['feature_audit']['train'] == audit['feature_audit']['validation'], 'Fit/inference schema mismatch')
            require(audit['learner_params'] == manifest['baseline_params'][branch], 'Learner settings changed')
    semantic = audits['no_semantics/S']
    require(len(semantic['removed_learner_columns']) == 33, 'No-semantics did not remove 33 columns')
    require(not any(c.startswith('semantic_') or c == 'historical_semantic_quality'
                    for c in semantic['learner_columns']), 'Semantic features retained')
    require(semantic['learner_params'] == manifest['baseline_params']['S'], 'No-semantics settings changed')
    raw = {}
    for column in components:
        if column.endswith('_raw'):
            # All B/S branches are XGBoost (float32); C is CatBoost (float64).
            dtype = 'float64' if column.endswith('_C_raw') else 'float32'
            raw[column] = components[column].to_numpy(dtype=dtype)
    for variant in ['checkpoint', 'full_refit', 'no_comparables', 'no_building']:
        expected_column = 'full' if variant == 'checkpoint' else variant
        got = reblend(*(raw[f'{variant}_{branch}_raw'] for branch in 'BCS'))
        require(np.array_equal(got, pred[expected_column].to_numpy()), f'CSV dtype-restored blend mismatch: {variant}')
    require(np.array_equal(reblend(raw['checkpoint_B_raw'], raw['checkpoint_C_raw'], raw['no_semantics_S_raw']),
                           pred.no_semantics.to_numpy()), 'No-semantics blend reconstruction failed')
    np.savez_compressed(folder / 'component_predictions_native.npz',
                        ml_num=components.ml_num.to_numpy(dtype=str), **raw)
    dtype_map = {key: str(value.dtype) for key, value in raw.items()}
    (folder / 'component_prediction_dtypes.json').write_text(json.dumps(dtype_map, indent=2) + '\n')

    names = ['no_comparables', 'no_building', 'no_semantics']
    labels = ['Remove comparables\n(all branches)', 'Remove building history\n(all branches)', 'Remove semantic features\n(S branch only)']
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={'width_ratios': [1.25, 1]})
    ax = axes[0]
    y = np.arange(3)
    effect = overall.loc[names, 'delta_MdAPE_pp'].to_numpy()
    for offset, lower, upper, color, label in [
        (-.07, 'paired_row_CI_low', 'paired_row_CI_high', '#2463a1', 'Paired row bootstrap'),
        (.07, 'paired_coordinate_cluster_CI_low', 'paired_coordinate_cluster_CI_high', '#c56b28', 'Coordinate-cluster bootstrap')]:
        lo = overall.loc[names, lower].to_numpy()
        hi = overall.loc[names, upper].to_numpy()
        ax.hlines(y + offset, lo, hi, color=color, lw=2, label=label)
        ax.plot(effect, y + offset, 'o', color=color, ms=5)
    ax.axvline(0, color='#777777', ls='--', lw=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel('Change in MdAPE vs full model (percentage points)\nPositive = removal hurts')
    ax.set_title('Matched feature ablations: effect with 95% CIs', loc='left', fontsize=12)
    ax.legend(loc='best', fontsize=8, frameon=False)
    ax.grid(axis='x', alpha=.2)
    ax = axes[1]
    member_names = ['full', 'incumbent_no_S', 'component_B', 'component_C', 'component_S']
    member_labels = ['Full blend', 'B + C blend', 'B: building XGB', 'C: CatBoost', 'S: semantic XGB']
    values = overall.loc[member_names, 'MdAPE'].to_numpy()
    bars = ax.barh(member_labels, values, color=['#2463a1', '#82a6c8', '#b9c6d0', '#b9c6d0', '#b9c6d0'])
    ax.invert_yaxis()
    for bar, value in zip(bars, values):
        ax.text(value + .025, bar.get_y() + bar.get_height()/2, f'{value:.3f}%', va='center', fontsize=9)
    ax.set_xlim(0, values.max() * 1.17)
    ax.set_xlabel('Median absolute percentage error (lower is better)')
    ax.set_title('Ensemble composition: same validation homes', loc='left', fontsize=12)
    ax.grid(axis='x', alpha=.2)
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle('OffMarketRefinedV6 | Matched fixed-origin fulltest | N = 6,047', fontsize=14, fontweight='bold')
    fig.text(.5, .015, 'Train: completed sales before Feb 1, 2026. Validation: Feb–May listing cohort with observed sales. Development evidence, not a fresh holdout.',
             ha='center', fontsize=8, color='#555555')
    fig.tight_layout(rect=[0, .07, 1, .93], w_pad=2)
    fig.savefig(folder / 'ablation_summary.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    verification = {'status': 'passed', 'validation_rows': len(pred), 'learner_fits': 10,
                    'checked': ['source artifact CSV hashes', 'row IDs and targets', 'all overall metrics',
                                'all segment metrics', 'feature counts and schemas', 'learner parameters',
                                'fresh full refit identity', 'all final feature-ablation blends from native-dtype raw arrays'],
                    'native_arrays_sha256': digest(folder / 'component_predictions_native.npz'),
                    'original_prediction_dtypes': dtype_map,
                    'note': 'CSV readers must restore B/S float32 and C float64 BEFORE taking logs; use NPZ for lossless native arrays.'}
    (folder / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    print(json.dumps({k:v for k,v in verification.items() if k != 'original_prediction_dtypes'}, indent=2))
    print(overall.loc[['full', *names], ['MdAPE', 'delta_MdAPE_pp', 'paired_row_CI_low', 'paired_row_CI_high',
                                        'paired_coordinate_cluster_CI_low', 'paired_coordinate_cluster_CI_high', 'MAPE']].to_string())


if __name__ == '__main__':
    main()
