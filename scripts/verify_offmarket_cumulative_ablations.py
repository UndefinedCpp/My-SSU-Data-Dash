"""Independently verify the cumulative experiment's saved artifacts (no training)."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def blend(b, c, s):
    incumbent = np.exp(.5 * (np.log(b) + np.log(c)))
    return np.round(np.exp(.75 * np.log(incumbent) + .25 * np.log(s)))


def metrics(pred, target):
    signed = 100 * (pred - target) / target
    ape = np.abs(signed)
    return {'N': len(target), 'MdAPE': float(np.median(ape)), 'MAPE': float(np.mean(ape)),
            'P90_APE': float(np.quantile(ape, .9)), 'median_signed_err': float(np.median(signed)),
            'mean_signed_err': float(np.mean(signed))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    out = args.run_dir.resolve()
    provenance = json.loads((out / 'provenance.json').read_text())
    require(provenance['status'] == 'completed' and provenance['new_learner_fits'] == 7, 'Incomplete run')
    for name, digest in provenance['artifact_sha256'].items():
        require(sha256(out / name) == digest, f'Hash mismatch: {name}')
    pred = pd.read_csv(out / 'predictions.csv', float_precision='round_trip')
    overall = pd.read_csv(out / 'overall_metrics.csv', float_precision='round_trip').set_index('variant')
    steps = pd.read_csv(out / 'sequential_metrics.csv', float_precision='round_trip')
    audit = json.loads((out / 'feature_audit.json').read_text())
    timing = pd.read_csv(out / 'timings.csv')
    raw = np.load(out / 'component_predictions_native.npz', allow_pickle=False)
    prior_dir = Path(provenance['prior_run'])
    require(sha256(prior_dir / 'provenance.json') == provenance['prior_provenance_sha256'], 'Prior manifest changed')
    prior = json.loads((prior_dir / 'provenance.json').read_text())
    require(prior['fresh_full_refit_matches_checkpoint'], 'No fresh baseline training control')
    prior_pred = pd.read_csv(prior_dir / 'predictions.csv', float_precision='round_trip')
    prior_raw = np.load(prior_dir / 'component_predictions_native.npz', allow_pickle=False)
    require(len(pred) == 6047 and pred.ml_num.is_unique, 'Wrong validation population')
    require(pred.ml_num.equals(prior_pred.ml_num) and pred.sold_price.equals(prior_pred.sold_price), 'Prior row/target mismatch')
    require(np.array_equal(raw['ml_num'], pred.ml_num.to_numpy()), 'Raw row IDs differ')
    require(np.array_equal(pred.full, prior_pred.full), 'Baseline changed')
    require(len(timing) == 7 and len(audit) == 7, 'Wrong fit/audit count')
    require(provenance['reused_branches'] == {'no_building_comparables_semantics':
            {'B': 'no_building_comparables/B', 'C': 'no_building_comparables/C'}}, 'Wrong reused branches')
    comparisons = {'full': tuple(prior_raw[f'checkpoint_{key}_raw'] for key in 'BCS'),
                   'no_building_comparables': tuple(raw[f'no_building_comparables_{key}_raw'] for key in 'BCS'),
                   'no_building_comparables_semantics': (raw['no_building_comparables_B_raw'], raw['no_building_comparables_C_raw'], raw['no_building_comparables_semantics_S_raw']),
                   'base_only': tuple(raw[f'base_only_{key}_raw'] for key in 'BCS')}
    target = pred.sold_price.to_numpy(float)
    for name, parts in comparisons.items():
        require([str(x.dtype) for x in parts] == ['float32', 'float64', 'float32'], 'Raw prediction dtype mismatch')
        require(np.array_equal(blend(*parts), pred[name].to_numpy()), f'Blend reconstruction mismatch: {name}')
        require(np.isfinite(pred[name]).all() and (pred[name] > 0).all(), 'Invalid prices')
        for key, value in metrics(pred[name].to_numpy(), target).items():
            require(np.isclose(value, overall.loc[name, key], rtol=0, atol=1e-10), f'Metric mismatch: {name}/{key}')
        require(np.isclose(overall.loc[name, 'delta_MdAPE_pp'], overall.loc[name, 'MdAPE']-overall.loc['full', 'MdAPE'], rtol=0, atol=1e-12), 'Baseline delta mismatch')
    for _, row in steps.iterrows():
        require(np.isclose(row.delta_MdAPE_pp, overall.loc[row.variant, 'MdAPE']-overall.loc[row.reference, 'MdAPE'], rtol=0, atol=1e-12), 'Sequential delta mismatch')
    for name, count in [('no_building_comparables/B', 217), ('no_building_comparables/C', 102),
                        ('no_building_comparables/S', 256), ('no_building_comparables_semantics/S', 223),
                        ('base_only/B', 44), ('base_only/C', 44), ('base_only/S', 44)]:
        record = audit[name]
        require(len(record['learner_columns']) == count, f'Wrong feature count: {name}')
        require(record['feature_audit']['train'] == record['feature_audit']['validation'], 'Fit/predict feature audit differs')
        branch = name[-1]
        expected_params = provenance['base_catboost_params'] if name == 'base_only/C' else provenance['baseline_params'][branch]
        require(record['learner_params'] == expected_params, f'Wrong learner settings: {name}')
        if name.startswith('base_only/'):
            require(record['learner_columns'] == provenance['base_columns'], 'Not literal base schema')
            require(sum(c.startswith('text:') for c in record['learner_columns']) == 5, 'Lost base text cues')
        else:
            columns = record['learner_columns']
            require(not any(c.startswith(('comp_', 'bldg_')) for c in columns), 'History features retained')
            if 'semantics/' in name:
                require(not any(c.startswith('semantic_') or c == 'historical_semantic_quality' for c in columns), 'Semantic features retained')
    base_params = provenance['base_catboost_params']
    require(base_params['cat_features'] == ['community'] and 'text_features' not in base_params and 'feature_calcers' not in base_params, 'Stale CatBoost input configuration')
    # Verify emitted Typst point estimates and deltas actually match the CSV.
    typst = (out / 'comparison.typ').read_text()
    for name, row in overall.iterrows():
        require(f'[{row.MdAPE:.4f}%]' in typst and f'[{row.MAPE:.4f}%]' in typst, 'Wrong Typst metric')
        if name != 'full':
            require(f'[{row.delta_MdAPE_pp:+.4f}]' in typst, 'Wrong Typst delta')
    verification = {'status': 'passed', 'validation_rows': len(pred), 'new_learner_fits': 7,
                    'checks': ['artifact hashes', 'same validation IDs/targets/order', 'prior matched baseline training control',
                               'all four blends from native-dtype arrays', 'all overall metrics', 'baseline and sequential deltas',
                               'exact fitted schemas/counts', 'base text cues retained', 'required CatBoost declarations only',
                               'learner settings', 'Typst point estimates'],
                    'typst_sha256': sha256(out / 'comparison.typ')}
    (out / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    # Context from earlier matched single-block removals: no extra fits.
    # These are separate from the requested four-row cumulative comparison.
    from scripts.run_offmarket_ablations import paired_ci
    groups = [np.asarray(pos) for pos in pred.groupby(['latitude', 'longitude'], dropna=False).indices.values()]
    context = []
    current_ape = np.abs(100 * (pred.no_building_comparables.to_numpy()-target) / target)
    for reference, question in [('no_building', 'Remove comparables when building block is already absent'),
                                ('no_comparables', 'Remove building block when comparables are already absent')]:
        reference_ape = np.abs(100 * (prior_pred[reference].to_numpy()-target) / target)
        row_ci = paired_ci(current_ape, reference_ape, 5000)
        cluster_ci = paired_ci(current_ape, reference_ape, 2000, groups=groups)
        context.append({'contrast': question, 'current': 'no_building_comparables', 'reference': reference,
                        'delta_MdAPE_pp': float(np.median(current_ape)-np.median(reference_ape)),
                        'paired_row_CI_low': row_ci[0], 'paired_row_CI_high': row_ci[1],
                        'paired_cluster_CI_low': cluster_ci[0], 'paired_cluster_CI_high': cluster_ci[1]})
    pd.DataFrame(context).to_csv(out / 'contextual_comparisons.csv', index=False)
    print(json.dumps(verification, indent=2))
    print(overall[['MdAPE', 'delta_MdAPE_pp', 'paired_row_CI_low', 'paired_row_CI_high', 'MAPE']].to_string())


if __name__ == '__main__':
    main()
