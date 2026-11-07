"""Matched fixed-origin OffMarketRefinedV6 ablations; production code unchanged.

Usage: python -u scripts/run_offmarket_ablations.py [--out-dir PATH]
Ten fits: full-model training reproducibility control (3), no-comps (3),
no-building (3), no-semantic S replacement (1). Component comparisons cost no fits.
Only load the trusted local checkpoint selected by --checkpoint.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import XGBoost before any optional Torch encoder to avoid OpenMP ordering issues.
import xgboost  # noqa: F401, E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data  # noqa: E402
from models.attempt1 import OffMarketBuildingXGBoostV1  # noqa: E402
from models.catboost_v2 import OffMarketCatBoostV2  # noqa: E402
from models.features import BUILDING_COLUMNS, COMP_COLUMNS  # noqa: E402
from models.robust_v6 import OffMarketRobustXGBV6  # noqa: E402
from models.semantic_v7 import OffMarketSemanticV7  # noqa: E402
from models.serialization import load_fitted_model  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / 'saved_models/fulltest/OffMarketRefinedV6_20261105_202825_072975.pkl'
CUTOFF = pd.Timestamp('2026-02-01')
BRANCHES = ('B', 'C', 'S')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class FeatureAudit:
    """Drop only named columns; retain upstream history/PCA and learner routines."""

    def __init__(self, drop_prefix=''):
        super().__init__()
        self.drop_prefix = drop_prefix
        self.feature_audit = {}

    def filter_features(self, frame, fit):
        require(frame.columns.is_unique, 'Duplicate feature columns')
        removed = [c for c in frame if self.drop_prefix and c.startswith(self.drop_prefix)]
        expected = list(COMP_COLUMNS if self.drop_prefix == 'comp_' else
                        BUILDING_COLUMNS if self.drop_prefix == 'bldg_' else ())
        require(set(removed) == set(expected), f'Wrong removed columns: {removed}')
        kept = frame.drop(columns=removed)
        info = {'input_columns': list(frame), 'removed_columns': removed,
                'output_columns': list(kept)}
        if fit:
            self.feature_audit['train'] = info
        else:
            require(info == self.feature_audit['train'], 'Train/inference feature schema differs')
            self.feature_audit['validation'] = info
        return kept


class AuditedBuildingXGB(FeatureAudit, OffMarketBuildingXGBoostV1):
    def _fit_features(self, homes):
        return self.filter_features(super()._fit_features(homes), True)

    def _transform_features(self, homes):
        return self.filter_features(super()._transform_features(homes), False)


class AuditedCatBoost(FeatureAudit, OffMarketCatBoostV2):
    def _features(self, homes, fit=False):
        return self.filter_features(super()._features(homes, fit=fit), fit)


class AuditedSemanticXGB(FeatureAudit, OffMarketSemanticV7):
    def _features(self, homes, fit=False):
        return self.filter_features(super()._features(homes, fit=fit), fit)


def learner_names(branch):
    if isinstance(branch, OffMarketCatBoostV2):
        return list(branch.model.feature_names_)
    return list(branch.model.get_booster().feature_names)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def dump(path, payload):
    path.write_text(json.dumps(clean(payload), indent=2, allow_nan=False) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def predict(branch, homes):
    values = branch.predict(homes)
    require(values.index.equals(homes.index), 'Prediction row index misalignment')
    # Keep native dtypes: production logs XGBoost's float32 prices before
    # mixing them with CatBoost float64. Upcasting first can change $1 rounding.
    result = values.to_numpy(copy=False)
    require(np.isfinite(result).all() and (result > 0).all(), 'Invalid component predictions')
    return result


def blend(parts):
    # Preserve the production nested floating-point computation before rounding.
    incumbent = np.exp(0.5 * (np.log(parts['B']) + np.log(parts['C'])))
    return np.exp(0.75 * np.log(incumbent) + 0.25 * np.log(parts['S']))


def metrics(prediction, target):
    signed = 100 * (prediction - target) / target
    ape = np.abs(signed)
    return {'N': len(target), 'MdAPE': float(np.median(ape)), 'MAPE': float(np.mean(ape)),
            'P90_APE': float(np.quantile(ape, .9)),
            'median_signed_err': float(np.median(signed)),
            'mean_signed_err': float(np.mean(signed))}


def paired_ci(ape, reference, replicates, seed=42, groups=None):
    """Percentile bootstrap of difference in pooled medians, not paired median."""
    rng = np.random.default_rng(seed)
    differences = np.empty(replicates)
    n = len(ape)
    for i in range(replicates):
        if groups is None:
            index = rng.integers(0, n, n)
        else:
            drawn = rng.integers(0, len(groups), len(groups))
            index = np.concatenate([groups[j] for j in drawn])
        differences[i] = np.median(ape[index]) - np.median(reference[index])
    return tuple(float(v) for v in np.quantile(differences, [.025, .975]))


def segment_masks(frame):
    yield 'overall', 'all', np.ones(len(frame), dtype=bool)
    for column in ('property_group', 'house_type_name', 'price_decile', 'listing_month',
                   'building_support', 'comparable_support'):
        for value in sorted(frame[column].dropna().unique()):
            yield column, str(value), frame[column].eq(value).to_numpy()


def markdown_table(frame, columns, decimals=4):
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return f'{value:.{decimals}f}'
        return str(value)
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    lines += ['| ' + ' | '.join(fmt(row[c]) for c in columns) + ' |'
              for _, row in frame.iterrows()]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--bootstrap', type=int, default=5000)
    parser.add_argument('--cluster-bootstrap', type=int, default=2000)
    args = parser.parse_args()
    require(args.bootstrap > 0 and args.cluster_bootstrap > 0, 'Bootstrap count must be positive')
    started = time.perf_counter()
    out = args.out_dir or ROOT / 'reports/ablations' / ('offmarket_refined_v6_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    log_handle = (out / 'run.log').open('w')

    def log(message):
        line = f'{datetime.now().isoformat(timespec="seconds")} {message}'
        print(line, flush=True)
        log_handle.write(line + '\n')
        log_handle.flush()

    log(f'Output directory: {out.resolve()}')
    checkpoint = args.checkpoint.resolve()
    metadata = json.loads(checkpoint.with_suffix('.json').read_text())
    require(metadata['model_id'] == 'OffMarketRefinedV6', 'Wrong model ID')
    require(metadata['cohort'] == 'date_listed', 'Wrong cohort')
    require(metadata['training_cutoff'] == CUTOFF.date().isoformat(), 'Wrong cutoff')
    with checkpoint.open('rb') as handle:
        baseline = load_fitted_model(handle)
    require(baseline.TRACK == 'off_market' and baseline.weight == .25, 'Wrong track/weight')
    require(type(baseline.incumbent).__name__ == 'OffMarketEnsembleV2', 'Wrong incumbent')
    original = {'B': baseline.incumbent.baseline, 'C': baseline.incumbent.candidate, 'S': baseline.candidate}
    require([type(original[k]).__name__ for k in BRANCHES] ==
            ['OffMarketBuildingXGBoostV1', 'OffMarketCatBoostV2', 'OffMarketSemanticV7'], 'Wrong branches')
    source_paths = [ROOT / 'data.py', ROOT / 'harness.py', Path(__file__).resolve(),
                    *sorted((ROOT / 'models').glob('*.py')), ROOT / 'models/refined_config.json']
    provenance = {'started_at': datetime.now().isoformat(), 'status': 'running',
                  'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
                  'checkpoint_metadata': metadata,
                  'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in source_paths},
                  'python': sys.version, 'platform': platform.platform(),
                  'packages': {p: importlib.metadata.version(p) for p in
                               ('xgboost', 'catboost', 'numpy', 'pandas', 'scikit-learn', 'scipy',
                                'sentence-transformers', 'torch', 'pyarrow')},
                  'blend_weights': {'B': .375, 'C': .375, 'S': .25},
                  'bootstrap': {'row_replicates': args.bootstrap, 'coordinate_cluster_replicates': args.cluster_bootstrap,
                                'seed': 42, 'statistic': 'median(APE_variant) - median(APE_reference)'},
                  'baseline_params': {k: original[k].model.get_params() for k in BRANCHES},
                  'baseline_learner_columns': {k: learner_names(original[k]) for k in BRANCHES}}
    listings = data.load('listings')
    train = listings[listings.date_sold < CUTOFF].copy()
    valid = listings[(listings.date_listed >= CUTOFF) & (listings.date_sold >= CUTOFF)
                     & listings.date_sold.notna()].copy()
    require((len(train), len(valid)) == (metadata['training_rows'], metadata['validation_rows']), 'Split size mismatch')
    require((len(train), len(valid)) == (142227, 6047), 'Unexpected data revision')
    require(train.ml_num.is_unique and valid.ml_num.is_unique, 'Duplicate listing IDs')
    require(set(train.ml_num).isdisjoint(valid.ml_num), 'Train/validation overlap')
    require(valid.date_listed.max() < pd.Timestamp('2026-06-01') and
            valid.date_sold.max() < pd.Timestamp('2026-06-01'), 'Data extends beyond frozen fulltest window')
    homes = data.hide_answers(valid, 'off_market')
    require(not set(data.AFTER_SALE_COLUMNS + ['list_price']).intersection(homes), 'Leaked query columns')
    target = valid.sold_price.to_numpy(float)
    require(np.isfinite(target).all() and (target > 0).all(), 'Invalid targets')
    train[['ml_num']].to_csv(out / 'training_ids.csv', index=False)
    frame = valid[['ml_num', 'sold_price', 'date_listed', 'date_sold', 'house_type_name', 'latitude', 'longitude']].reset_index(drop=True)
    frame['property_group'] = np.select(
        [frame.house_type_name.isin(['Condo Apt', 'Condo Townhouse']),
         frame.house_type_name.isin(['Detached', 'Semi-Detached', 'Freehold Townhouse'])],
        ['condos', 'houses'], default='other')
    frame['price_decile'] = pd.qcut(frame.sold_price, 10, labels=[f'D{i}' for i in range(1, 11)])
    frame['listing_month'] = frame.date_listed.dt.strftime('%Y-%m')
    building = original['B'].building.transform(homes)
    comps = original['B'].comps.transform(homes)
    frame['bldg_n'] = building.bldg_n.to_numpy()
    frame['comp_count'] = comps.comp_count.to_numpy()
    frame['comp_effective_n'] = comps.comp_effective_n.to_numpy()
    frame['building_support'] = np.where(frame.bldg_n.fillna(0) > 0, 'prior_building_sales', 'no_prior_building_sales')
    eff = frame.comp_effective_n.fillna(0).to_numpy()
    frame['comparable_support'] = np.select([eff <= 0, eff < 10, eff < 25],
                                            ['none', 'low_0_to_10', 'medium_10_to_25'], default='high_25_plus')
    provenance.update({'training_rows': len(train), 'validation_rows': len(valid),
                       'data_path': str(Path(data.find_file('listings')).resolve()),
                       'data_sha256': sha256(Path(data.find_file('listings'))),
                       'training_ids_sha256': sha256(out / 'training_ids.csv'),
                       'validation_id_order_sha256': hashlib.sha256('\n'.join(frame.ml_num).encode()).hexdigest(),
                       'train_max_sold': str(train.date_sold.max()),
                       'validation_min_listed': str(valid.date_listed.min()),
                       'validation_max_listed': str(valid.date_listed.max()),
                       'validation_max_sold': str(valid.date_sold.max()),
                       'support_definitions': 'Building n>0; comp effective n bins: none, (0,10), [10,25), >=25. Fixed full-model support.'})
    dump(out / 'provenance.json', provenance)
    component_predictions = frame[['ml_num', 'sold_price']].copy()
    prediction_columns = []
    details = {}
    timings = []

    def save_prediction(name, raw):
        require(len(raw) == len(frame) and np.isfinite(raw).all() and (raw > 0).all(), f'Invalid {name}')
        frame[name] = np.round(raw)
        prediction_columns.append(name)
        frame.to_csv(out / 'predictions.csv', index=False)
        log(f'{name}: {json.dumps(metrics(frame[name].to_numpy(), target))}')

    log(f'Matched split: train={len(train)}, validation={len(valid)}')
    original_predictions = {}
    for key in BRANCHES:
        tick = time.perf_counter()
        original_predictions[key] = predict(original[key], homes)
        component_predictions[f'checkpoint_{key}_raw'] = original_predictions[key]
        timings.append({'variant': 'checkpoint', 'branch': key, 'fit_seconds': 0,
                        'predict_seconds': time.perf_counter() - tick})
    direct = predict(baseline, homes)
    require(np.array_equal(np.round(direct), np.round(blend(original_predictions))), 'Nested blend mismatch')
    save_prediction('full', direct)
    observed = metrics(frame['full'].to_numpy(), target)
    for key in ('N', 'MdAPE', 'MAPE', 'median_signed_err', 'mean_signed_err'):
        require(np.isclose(observed[key], metadata['metrics'][key], rtol=0, atol=1e-9), f'Baseline does not reproduce {key}')
    provenance['checkpoint_reproduction'] = observed
    dump(out / 'provenance.json', provenance)
    log('Saved checkpoint baseline reproduced to absolute metric tolerance 1e-9')
    for key in BRANCHES:
        save_prediction(f'component_{key}', original_predictions[key])
    save_prediction('incumbent_no_S', np.exp(.5 * (np.log(original_predictions['B']) + np.log(original_predictions['C']))))
    save_prediction('no_B_reweighted', np.exp(.6 * np.log(original_predictions['C']) + .4 * np.log(original_predictions['S'])))
    save_prediction('no_C_reweighted', np.exp(.6 * np.log(original_predictions['B']) + .4 * np.log(original_predictions['S'])))
    component_predictions.to_csv(out / 'component_predictions.csv', index=False)

    def fit_branch(variant, key, branch, removed_prefix):
        log(f'START fit {variant}/{key}')
        expected_params = clean(original[key].model.get_params())
        if key == 'B':
            require(clean(branch.model.get_params()) == expected_params, 'B learner parameters changed')
        tick = time.perf_counter()
        branch.train(train)
        fit_seconds = time.perf_counter() - tick
        require(clean(branch.model.get_params()) == expected_params, f'{variant}/{key}: learner settings changed')
        baseline_names = learner_names(original[key])
        expected_names = [name for name in baseline_names if not (
            name.startswith(removed_prefix) if removed_prefix else False)]
        if variant == 'no_semantics':
            expected_names = [name for name in baseline_names if not
                              (name.startswith('semantic_') or name == 'historical_semantic_quality')]
        actual_names = learner_names(branch)
        require(actual_names == expected_names, f'{variant}/{key}: learner feature schema mismatch')
        tick = time.perf_counter()
        raw = predict(branch, homes)
        predict_seconds = time.perf_counter() - tick
        if isinstance(branch, FeatureAudit):
            require('validation' in branch.feature_audit, 'Missing inference audit')
        details[f'{variant}/{key}'] = {'learner_params': branch.model.get_params(),
                                     'learner_columns': actual_names,
                                     'removed_learner_columns': [c for c in baseline_names if c not in actual_names],
                                     'feature_audit': getattr(branch, 'feature_audit', None),
                                     'metrics': metrics(np.round(raw), target)}
        if variant == 'full_refit':
            require(np.array_equal(np.round(raw), np.round(original_predictions[key])),
                    f'Fresh full {key} refit does not reproduce checkpoint rounded predictions; stop before ablations')
            details[f'{variant}/{key}']['maximum_unrounded_difference_from_checkpoint'] = float(np.max(np.abs(raw-original_predictions[key])))
        timings.append({'variant': variant, 'branch': key, 'fit_seconds': fit_seconds,
                        'predict_seconds': predict_seconds})
        component_predictions[f'{variant}_{key}_raw'] = raw
        component_predictions.to_csv(out / 'component_predictions.csv', index=False)
        dump(out / 'feature_audit.json', details)
        pd.DataFrame(timings).to_csv(out / 'timings.csv', index=False)
        log(f'DONE {variant}/{key}: fit {fit_seconds:.2f}s, predict {predict_seconds:.2f}s, '
            f'features={len(actual_names)}, MdAPE={metrics(np.round(raw), target)["MdAPE"]:.6f}')
        return raw

    for variant, prefix in [('full_refit', ''), ('no_comparables', 'comp_'), ('no_building', 'bldg_')]:
        parts = {}
        for key, cls in zip(BRANCHES, (AuditedBuildingXGB, AuditedCatBoost, AuditedSemanticXGB)):
            branch = cls(prefix)
            parts[key] = fit_branch(variant, key, branch, prefix)
            del branch
            gc.collect()
        save_prediction(variant, blend(parts))
        if variant == 'full_refit':
            require(np.array_equal(frame.full.to_numpy(), frame.full_refit.to_numpy()), 'Full refit blend mismatch')
            provenance['fresh_full_refit_matches_checkpoint'] = True
            dump(out / 'provenance.json', provenance)
            log('Fresh full-model training reproduces every checkpoint rounded component and blend prediction')
    parent = OffMarketRobustXGBV6()
    parts = dict(original_predictions)
    parts['S'] = fit_branch('no_semantics', 'S', parent, '')
    save_prediction('no_semantics', blend(parts))
    del parent
    component_predictions.to_csv(out / 'component_predictions.csv', index=False)

    log('All ten fits finished; computing paired uncertainty and fixed-segment summaries')
    groups = [np.asarray(pos) for pos in frame.groupby(['latitude', 'longitude'], dropna=False).indices.values()]
    reference = 100 * np.abs(frame.full.to_numpy() - target) / target
    rows = []
    for name in prediction_columns:
        row = {'variant': name, **metrics(frame[name].to_numpy(), target)}
        ape = 100 * np.abs(frame[name].to_numpy() - target) / target
        row['delta_MdAPE_pp'] = row['MdAPE'] - float(np.median(reference))
        if np.array_equal(ape, reference):
            row_ci = cluster_ci = (0., 0.)
        else:
            row_ci = paired_ci(ape, reference, args.bootstrap)
            cluster_ci = paired_ci(ape, reference, args.cluster_bootstrap, groups=groups)
        row.update({'paired_row_CI_low': row_ci[0], 'paired_row_CI_high': row_ci[1],
                    'paired_coordinate_cluster_CI_low': cluster_ci[0], 'paired_coordinate_cluster_CI_high': cluster_ci[1]})
        rows.append(row)
    results = pd.DataFrame(rows)
    results.to_csv(out / 'overall_metrics.csv', index=False)
    segment_rows = []
    for dimension, label, mask in segment_masks(frame):
        baseline_mdape = metrics(frame.full.to_numpy()[mask], target[mask])['MdAPE']
        for name in prediction_columns:
            row = {'variant': name, 'dimension': dimension, 'segment': label,
                   **metrics(frame[name].to_numpy()[mask], target[mask])}
            row['delta_MdAPE_pp'] = row['MdAPE'] - baseline_mdape
            segment_rows.append(row)
    segments = pd.DataFrame(segment_rows)
    segments.to_csv(out / 'segment_metrics.csv', index=False)
    branch_rows = []
    for variant in ('checkpoint', 'full_refit', 'no_comparables', 'no_building', 'no_semantics'):
        for key in BRANCHES:
            column = f'{variant}_{key}_raw'
            if column not in component_predictions:
                continue
            summary = metrics(np.round(component_predictions[column].to_numpy()), target)
            ref_mdape = metrics(np.round(original_predictions[key]), target)['MdAPE']
            branch_rows.append({'variant': variant, 'branch': key, **summary,
                                'delta_MdAPE_vs_same_full_branch_pp': summary['MdAPE'] - ref_mdape})
    branch_results = pd.DataFrame(branch_rows)
    branch_results.to_csv(out / 'branch_metrics.csv', index=False)
    pd.DataFrame(timings).to_csv(out / 'timings.csv', index=False)
    require(all(sha256(ROOT / p) == digest for p, digest in provenance['source_sha256'].items()), 'Source changed during experiment')
    provenance.update({'status': 'completed', 'finished_at': datetime.now().isoformat(),
                       'total_seconds': time.perf_counter()-started, 'additional_learner_fits': 10,
                       'ablation_learner_fits': 7, 'training_control_learner_fits': 3,
                       'coordinate_clusters': len(groups),
                       'artifacts_sha256': {p.name: sha256(p) for p in out.glob('*.csv')}})
    dump(out / 'provenance.json', provenance)
    main_names = ['full', 'no_comparables', 'no_building', 'no_semantics']
    columns = ['variant', 'N', 'MdAPE', 'delta_MdAPE_pp', 'paired_row_CI_low', 'paired_row_CI_high',
               'MAPE', 'P90_APE', 'median_signed_err']
    main_table = markdown_table(results[results.variant.isin(main_names)], columns)
    component_table = markdown_table(results[results.variant.isin(
        ['component_B', 'component_C', 'component_S', 'incumbent_no_S', 'no_B_reweighted', 'no_C_reweighted'])],
        ['variant', 'MdAPE', 'delta_MdAPE_pp', 'MAPE', 'median_signed_err'])
    selected_segments = segments[segments.variant.isin(main_names) & segments.dimension.isin(
        ['property_group', 'building_support', 'comparable_support', 'listing_month'])]
    segment_table = markdown_table(selected_segments, ['dimension', 'segment', 'variant', 'N', 'MdAPE',
                                                      'delta_MdAPE_pp', 'median_signed_err'])
    report = f'''# OffMarketRefinedV6 matched fulltest ablation results

## Protocol and baseline verification

- Train on {len(train):,} completed sales before 2026-02-01. Score the exact {len(valid):,} fulltest listing-date validation rows from February–May 2026.
- Saved baseline reproduces MdAPE/MAPE/signed errors to 1e-9 absolute tolerance.
- Fresh full-model refit reproduces **every rounded validation component prediction and final blend prediction**. Its three fits are reproducibility controls, not extra ablations.
- B/C/S weights remain 0.375/0.375/0.25 in nested log-price blending. Blend unrounded component prices and round only the final output.
- Global block removals refit all three affected learners. No-semantics refits the S parent learner with the same settings, retaining lexical text and B/C.
- Feature schemas, removed columns, learner-parameter equality, row identity, and hidden query answers are asserted. No production model/configuration edits.

## Matched feature ablations

Positive delta means the removed block helped the full model. All errors are percentages; deltas and confidence limits are percentage points.

{main_table}

Paired row-bootstrap intervals use {args.bootstrap:,} replicates (seed 42), resampling the same rows for both pooled medians. Exact-coordinate cluster-bootstrap sensitivity uses {args.cluster_bootstrap:,} replicates across {len(groups):,} clusters, retaining all rows per sampled cluster; its intervals are in [overall metrics](<overall_metrics.csv>). These intervals do not adjust for temporal dependence, multiple comparisons, or development selection.

## Ensemble composition (no training)

B = building XGBoost, C = CatBoost, S = semantic XGBoost. These rows compare different learners/features, not pure semantic feature attribution. No-B and no-C renormalize the remaining original weights to 0.6/0.4.

{component_table}

## Support and temporal breakdown

Groups are fixed using the full-model history features. Condo group includes Condo Apt and Condo Townhouse; houses includes Detached, Semi-Detached and Freehold Townhouse; other includes Co-Op Apt. Comparable support uses effective neighbor count, not raw sales count.

{segment_table}

See [all segment results](<segment_metrics.csv>) for individual house types and fixed sold-price deciles. Sold-price bins are retrospective diagnostic groups, never prediction inputs.

## Interpretation limits

- No-comparables retains exact-building history. No-building retains nearby comparables, which may include same-building sales. These are conditional block contributions, not removal of every form of historical price information.
- No-semantics is **not no-text**: fixed text cues/terms and CatBoost BoW remain.
- Feature filtering deliberately retains the upstream feature-computation paths. Timings measure this experiment, not optimized inference speed after removing blocks.
- The validation window has already been used for development/model selection. Results are development ablation evidence, not an unbiased hidden-test estimate. The cohort contains observed completed sales, not all future listings.
- No variant is automatically adopted; the submission model remains unchanged.

## Reproducibility artifacts

[Predictions with IDs, targets and support groups](<predictions.csv>) · [unrounded branch predictions](<component_predictions.csv>) · [overall metrics and paired intervals](<overall_metrics.csv>) · [branch metrics](<branch_metrics.csv>) · [feature/parameter audits](<feature_audit.json>) · [timings](<timings.csv>) · [source/data/checkpoint fingerprints and environment](<provenance.json>) · [run log](<run.log>) · [training IDs](<training_ids.csv>).

Completed in {provenance['total_seconds'] / 60:.2f} minutes; ten learner fits including the three-fit training control.
'''
    (out / 'summary.md').write_text(report)
    log(f'COMPLETE: {out / "summary.md"}')
    print(main_table, flush=True)
    log_handle.close()


if __name__ == '__main__':
    main()
