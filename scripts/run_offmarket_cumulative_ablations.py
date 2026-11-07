"""Run cumulative fulltest ablations, including literal base_features()-only.

Run from project root: python -u scripts/run_offmarket_cumulative_ablations.py
Reuses the earlier verified baseline and fits seven new learners. B/C are shared
between the two joint-history-removal variants because their inputs are identical.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_offmarket_ablations import (  # imports XGBoost first
    BRANCHES, CUTOFF, DEFAULT_CHECKPOINT, blend, clean, dump, learner_names,
    markdown_table, metrics, paired_ci, predict, require, sha256,
)
import numpy as np
import pandas as pd
import data
from models.attempt1 import OffMarketBuildingXGBoostV1, _as_community_categories
from models.catboost_v2 import OffMarketCatBoostV2
from models.features import BUILDING_COLUMNS, COMP_COLUMNS, base_features
from models.robust_v6 import OffMarketRobustXGBV6
from models.semantic_v7 import OffMarketSemanticV7
from models.serialization import StableCatBoostRegressor, load_fitted_model

PRIOR_RUN = ROOT / 'reports/ablations/offmarket_refined_v6_20261107_025626'
JOINT_COLUMNS = set(BUILDING_COLUMNS) | set(COMP_COLUMNS)
VARIANTS = ['full', 'no_building_comparables', 'no_building_comparables_semantics', 'base_only']
LABELS = {
    'full': 'Full baseline',
    'no_building_comparables': 'Baseline − building & comparables',
    'no_building_comparables_semantics': 'Baseline − building & comparables & semantics',
    'base_only': 'Only base_features()',
}


class Audit:
    def audit(self, source, output, fit, expected_removed=None):
        require(source.columns.is_unique and output.columns.is_unique, 'Duplicate feature columns')
        removed = [c for c in source if c not in output]
        if expected_removed is not None:
            require(set(removed) == set(expected_removed), 'Unexpected feature removal')
        record = {'input_columns': list(source), 'output_columns': list(output), 'removed_columns': removed}
        if fit:
            self.feature_audit = {'train': record}
        else:
            require(record == self.feature_audit['train'], 'Train/inference schema mismatch')
            self.feature_audit['validation'] = record
        return output

    def remove_history(self, source, fit):
        return self.audit(source, source.drop(columns=list(JOINT_COLUMNS)), fit, JOINT_COLUMNS)


class JointBuilding(Audit, OffMarketBuildingXGBoostV1):
    def _fit_features(self, homes):
        return self.remove_history(super()._fit_features(homes), True)

    def _transform_features(self, homes):
        return self.remove_history(super()._transform_features(homes), False)


class JointCatBoost(Audit, OffMarketCatBoostV2):
    def _features(self, homes, fit=False):
        return self.remove_history(super()._features(homes, fit=fit), fit)


class JointSemantic(Audit, OffMarketSemanticV7):
    def _features(self, homes, fit=False):
        return self.remove_history(super()._features(homes, fit=fit), fit)


class JointRobust(Audit, OffMarketRobustXGBV6):
    def _features(self, homes, fit=False):
        return self.remove_history(super()._features(homes, fit=fit), fit)


class LiteralBase(Audit):
    def literal_base(self, homes, fit, fill_numeric=False):
        raw = base_features(homes, list_aware=False)
        frame = raw.copy()
        if fill_numeric:
            numeric = frame.select_dtypes(include='number').columns
            frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan).fillna(-999)
        return self.audit(raw, frame, fit, ())


class BaseBuilding(LiteralBase, OffMarketBuildingXGBoostV1):
    def _fit_features(self, homes):
        frame = self.literal_base(homes, True)
        frame, self.community_categories_ = _as_community_categories(frame)
        return frame

    def _transform_features(self, homes):
        require(self.community_categories_ is not None, 'Fit first')
        frame, _ = _as_community_categories(self.literal_base(homes, False), self.community_categories_)
        return frame


class BaseCatBoost(LiteralBase, OffMarketCatBoostV2):
    def __init__(self, learner_params):
        super().__init__()
        self.learner_params = dict(learner_params)

    def _features(self, homes, fit=False):
        return self.literal_base(homes, fit, fill_numeric=True)

    def _train(self, homes):
        frame = self._features(homes, fit=True)
        self.model = StableCatBoostRegressor(**self.learner_params)
        self.model.fit(frame, np.log(homes.sold_price.to_numpy(float)))
        return self


class BaseRobust(LiteralBase, OffMarketRobustXGBV6):
    def _features(self, homes, fit=False):
        return self.literal_base(homes, fit, fill_numeric=True)

    def _numeric(self, frame, fit=False):
        # Parent drops raw categorical/text columns absent from literal base().
        # Preserve its train-fitted native community categorical treatment.
        frame = frame.copy()
        if fit:
            self.community_categories_ = sorted(frame.community.unique())
        frame['community'] = pd.Categorical(
            frame.community.where(frame.community.isin(self.community_categories_)),
            categories=self.community_categories_,
        )
        return frame


def adapted_base_cat_params(params):
    params = dict(params)
    params['cat_features'] = ['community']
    params.pop('text_features', None)
    params.pop('feature_calcers', None)
    return params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-run', type=Path, default=PRIOR_RUN)
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args()
    prior_dir = args.prior_run.resolve()
    prior = json.loads((prior_dir / 'provenance.json').read_text())
    require(prior['status'] == 'completed' and prior['fresh_full_refit_matches_checkpoint'], 'Unverified prior baseline')
    # The prior fresh training control remains valid only with identical sources,
    # data, checkpoint, split and dependency versions.
    for rel, digest in prior['source_sha256'].items():
        require(sha256(ROOT / rel) == digest, f'Source changed since full-refit control: {rel}')
    for package, version in prior['packages'].items():
        require(importlib.metadata.version(package) == version, f'Dependency changed: {package}')
    require(sha256(Path(data.find_file('listings'))) == prior['data_sha256'], 'Training data changed')
    checkpoint = Path(prior['checkpoint'])
    require(checkpoint == DEFAULT_CHECKPOINT and sha256(checkpoint) == prior['checkpoint_sha256'], 'Checkpoint changed')
    out = args.out_dir or ROOT / 'reports/ablations' / ('offmarket_cumulative_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    log_file = (out / 'run.log').open('w')
    started = time.perf_counter()
    provenance = {'status': 'running', 'started_at': datetime.now().isoformat(),
                  'prior_run': str(prior_dir), 'prior_provenance_sha256': sha256(prior_dir / 'provenance.json'),
                  'prior_fresh_full_refit_matches_checkpoint': True,
                  'checkpoint_sha256': prior['checkpoint_sha256'], 'data_sha256': prior['data_sha256'],
                  'packages': prior['packages'], 'baseline_params': prior['baseline_params'],
                  'source_sha256': {**prior['source_sha256'], str(Path(__file__).resolve().relative_to(ROOT)): sha256(Path(__file__).resolve())},
                  'blend_weights': prior['blend_weights'],
                  'bootstrap': prior['bootstrap'],
                  'base_only_definition': 'Exactly literal base_features(list_aware=False) columns, including five text cues. Branch-native missing-value/category preprocessing retained. C categorical declaration reduced to community; native BoW text config removed.',
                  'base_catboost_params': adapted_base_cat_params(prior['baseline_params']['C'])}
    dump(out / 'provenance.json', provenance)

    def log(message):
        text = f'{datetime.now().isoformat(timespec="seconds")} {message}'
        print(text, flush=True)
        log_file.write(text + '\n')
        log_file.flush()

    try:
        log(f'Output directory: {out}')
        listings = data.load('listings')
        train = listings[listings.date_sold < CUTOFF].copy()
        valid = listings[(listings.date_listed >= CUTOFF) & (listings.date_sold >= CUTOFF) & listings.date_sold.notna()].copy()
        frame = pd.read_csv(prior_dir / 'predictions.csv', float_precision='round_trip')
        require((len(train), len(valid)) == (142227, 6047), 'Split size mismatch')
        require(valid.ml_num.is_unique and train.ml_num.is_unique, 'Duplicate IDs')
        require(set(valid.ml_num).isdisjoint(train.ml_num), 'Train/valid overlap')
        require(np.array_equal(valid.ml_num.to_numpy(), frame.ml_num.to_numpy()), 'Validation IDs/order changed')
        require(np.array_equal(valid.sold_price.to_numpy(), frame.sold_price.to_numpy()), 'Targets changed')
        require(sha256(prior_dir / 'training_ids.csv') == prior['training_ids_sha256'], 'Prior training manifest changed')
        expected_train_ids = pd.read_csv(prior_dir / 'training_ids.csv').ml_num.to_numpy()
        require(np.array_equal(train.ml_num.to_numpy(), expected_train_ids), 'Training IDs/order changed')
        homes = data.hide_answers(valid, 'off_market')
        require(not set(data.AFTER_SALE_COLUMNS + ['list_price']).intersection(homes), 'Leaked query answers')
        target = valid.sold_price.to_numpy(float)
        with checkpoint.open('rb') as handle:
            baseline = load_fitted_model(handle)
        direct = predict(baseline, homes)
        require(np.array_equal(np.round(direct), frame.full.to_numpy()), 'Checkpoint no longer reproduces prior predictions')
        require(baseline.weight == .25, 'Blend weight changed')
        originals = {'B': baseline.incumbent.baseline, 'C': baseline.incumbent.candidate, 'S': baseline.candidate}
        require([type(originals[k]).__name__ for k in BRANCHES] == ['OffMarketBuildingXGBoostV1', 'OffMarketCatBoostV2', 'OffMarketSemanticV7'], 'Branch types changed')
        for key in BRANCHES:
            require(clean(originals[key].model.get_params()) == prior['baseline_params'][key], 'Baseline learner params changed')
        base_columns = list(base_features(homes))
        require(len(base_columns) == 44 and sum(c.startswith('text:') for c in base_columns) == 5, 'Unexpected literal base feature definition')
        provenance.update({'training_rows': len(train), 'validation_rows': len(valid), 'base_columns': base_columns,
                           'baseline_reproduction': metrics(np.round(direct), target),
                           'validation_id_order_sha256': prior['validation_id_order_sha256'],
                           'training_ids_sha256': prior['training_ids_sha256']})
        dump(out / 'provenance.json', provenance)
        previous_prediction_columns = ['component_B', 'component_C', 'component_S', 'incumbent_no_S',
                                       'no_B_reweighted', 'no_C_reweighted', 'full_refit',
                                       'no_comparables', 'no_building', 'no_semantics']
        frame = frame.drop(columns=[c for c in previous_prediction_columns if c in frame]).copy()
        audit = {}
        timings = []
        raw_arrays = {}
        branches = []

        def fit_branch(variant, key, branch, semantic_removed=False, base=False):
            log(f'START {variant}/{key}')
            tick = time.perf_counter()
            branch.train(train)
            fit_seconds = time.perf_counter() - tick
            expected_params = prior['baseline_params'][key]
            if base and key == 'C':
                expected_params = provenance['base_catboost_params']
            require(clean(branch.model.get_params()) == expected_params, f'Learner params changed: {variant}/{key}')
            expected_names = base_columns if base else [c for c in learner_names(originals[key]) if c not in JOINT_COLUMNS and not (semantic_removed and (c.startswith('semantic_') or c == 'historical_semantic_quality'))]
            require(learner_names(branch) == expected_names, f'Learner columns changed: {variant}/{key}')
            tick = time.perf_counter()
            raw = predict(branch, homes)
            prediction_seconds = time.perf_counter() - tick
            require(branch.feature_audit['train'] == branch.feature_audit['validation'], 'Feature audit mismatch')
            actual_removed = [c for c in learner_names(originals[key]) if c not in expected_names]
            audit[f'{variant}/{key}'] = {'learner_params': branch.model.get_params(), 'learner_columns': expected_names,
                                       'removed_learner_columns': actual_removed, 'feature_audit': branch.feature_audit,
                                       'prediction_dtype': str(raw.dtype)}
            name = f'{variant}_{key}_raw'
            raw_arrays[name] = raw
            branches.append({'variant': variant, 'branch': key, **metrics(np.round(raw), target)})
            timings.append({'variant': variant, 'branch': key, 'fit_seconds': fit_seconds, 'predict_seconds': prediction_seconds})
            dump(out / 'feature_audit.json', audit)
            pd.DataFrame(timings).to_csv(out / 'timings.csv', index=False)
            np.savez_compressed(out / 'component_predictions_native.npz', ml_num=frame.ml_num.to_numpy(dtype=str), **raw_arrays)
            log(f'DONE {variant}/{key}: fit={fit_seconds:.2f}s, features={len(expected_names)}, MdAPE={metrics(np.round(raw), target)["MdAPE"]:.6f}')
            return raw

        def save_blend(variant, parts):
            frame[variant] = np.round(blend(parts))
            frame.to_csv(out / 'predictions.csv', index=False)
            log(f'{variant}: {json.dumps(metrics(frame[variant].to_numpy(), target))}')

        parts = {}
        for key, cls in zip(BRANCHES, (JointBuilding, JointCatBoost, JointSemantic)):
            branch = cls()
            parts[key] = fit_branch('no_building_comparables', key, branch)
            del branch
            gc.collect()
        save_blend('no_building_comparables', parts)
        branch = JointRobust()
        parts['S'] = fit_branch('no_building_comparables_semantics', 'S', branch, semantic_removed=True)
        del branch
        gc.collect()
        save_blend('no_building_comparables_semantics', parts)
        provenance['reused_branches'] = {'no_building_comparables_semantics': {'B': 'no_building_comparables/B', 'C': 'no_building_comparables/C'}}
        parts = {}
        for key, branch in [('B', BaseBuilding()), ('C', BaseCatBoost(provenance['base_catboost_params'])), ('S', BaseRobust())]:
            parts[key] = fit_branch('base_only', key, branch, base=True)
            del branch
            gc.collect()
        save_blend('base_only', parts)
        pd.DataFrame(branches).to_csv(out / 'branch_metrics.csv', index=False)
        pd.DataFrame(timings).to_csv(out / 'timings.csv', index=False)
        log('Seven fits completed. Computing paired row and coordinate-cluster intervals.')
        groups = [np.asarray(pos) for pos in frame.groupby(['latitude', 'longitude'], dropna=False).indices.values()]
        base_ape = np.abs(100 * (frame.full.to_numpy()-target) / target)
        rows = []
        sequential = []
        for index, name in enumerate(VARIANTS):
            ape = np.abs(100 * (frame[name].to_numpy()-target) / target)
            row = {'variant': name, **metrics(frame[name].to_numpy(), target)}
            row['delta_MdAPE_pp'] = row['MdAPE']-float(np.median(base_ape))
            ci = paired_ci(ape, base_ape, 5000) if index else (0., 0.)
            cluster = paired_ci(ape, base_ape, 2000, groups=groups) if index else (0., 0.)
            row.update({'paired_row_CI_low': ci[0], 'paired_row_CI_high': ci[1], 'paired_cluster_CI_low': cluster[0], 'paired_cluster_CI_high': cluster[1]})
            rows.append(row)
            if index:
                previous = VARIANTS[index-1]
                ref = np.abs(100 * (frame[previous].to_numpy()-target) / target)
                step_ci = paired_ci(ape, ref, 5000)
                step_cluster = paired_ci(ape, ref, 2000, groups=groups)
                sequential.append({'variant': name, 'reference': previous, 'delta_MdAPE_pp': float(np.median(ape)-np.median(ref)),
                                   'paired_row_CI_low': step_ci[0], 'paired_row_CI_high': step_ci[1],
                                   'paired_cluster_CI_low': step_cluster[0], 'paired_cluster_CI_high': step_cluster[1]})
        results = pd.DataFrame(rows)
        results.to_csv(out / 'overall_metrics.csv', index=False)
        pd.DataFrame(sequential).to_csv(out / 'sequential_metrics.csv', index=False)
        require(all(sha256(ROOT / rel) == digest for rel, digest in provenance['source_sha256'].items()), 'Source changed during run')
        provenance.update({'status': 'completed', 'finished_at': datetime.now().isoformat(), 'new_learner_fits': 7,
                           'total_seconds': time.perf_counter()-started,
                           'raw_prediction_dtypes': {name: str(raw.dtype) for name, raw in raw_arrays.items()},
                           'coordinate_clusters': len(groups),
                           'artifact_sha256': {p.name: sha256(p) for p in out.iterdir() if p.suffix in ('.csv', '.npz')}})
        dump(out / 'provenance.json', provenance)
        table = markdown_table(results, ['variant', 'MdAPE', 'delta_MdAPE_pp', 'paired_row_CI_low', 'paired_row_CI_high', 'MAPE', 'P90_APE'])
        (out / 'summary.md').write_text(f'''# Cumulative OffMarketRefinedV6 ablation comparison

## Matched fulltest results

Same 142,227 training / 6,047 validation homes as the verified fixed-origin fulltest; training sales before 2026-02-01 and February–May listing-date validation. Checkpoint predictions match the prior full-model refit exactly; source/data/dependency fingerprints are unchanged. Learner settings and log-blend weights remain B=0.375, C=0.375, S=0.25, except required base-only CatBoost input declarations.

{table}

Deltas and confidence limits are percentage points relative to full baseline. Positive means removal hurts. Paired row-bootstrap intervals use 5,000 replicates; [coordinate-cluster intervals](<overall_metrics.csv>) use 2,000 replicates across {len(groups):,} exact-coordinate clusters. [Sequential comparisons](<sequential_metrics.csv>) estimate each additional cumulative removal against the preceding row, not just full baseline.

## Exact configurations

1. **Full baseline:** unchanged production OffMarketRefinedV6.
2. **Baseline − building & comparables:** remove all 24 building and 14 comparable columns from each branch. B/S spline/tensor features, fixed text cues/terms, C BoW remarks, and S semantic channels remain.
3. **Baseline − building & comparables & semantics:** additionally remove S's 32 PCA embedding dimensions and historical semantic-quality column, using the same robust parent learner. B/C are identical to row 2, so their fitted predictions are reused. Other lexical text and spline features remain.
4. **Only literal base_features():** all three learners receive exactly the same 44 base columns, including five text-cue counts, with branch-native numeric/category preprocessing. No building, comparables, spline/tensor features, fixed 47-term block, native CatBoost BoW, semantic channels, or enriched branch-specific structural columns. CatBoost uses community as its only categorical feature and has no native text configuration; all other learner settings remain unchanged. This row is not text-free.

Base-only is a broader reduction than just removing semantics from row 3: it also removes additional engineered features and native categorical/text representations. The cumulative differences therefore are not uniquely additive feature contributions. All comparisons use fixed deployed seeds; feature deletion can change XGBoost feature-sampling draws. Intervals do not account for development selection, temporal dependence, or multiple testing. This validation window is not a fresh holdout. Production model/configuration is unchanged; no candidate is adopted automatically.

## Artifacts

[Final predictions with IDs/targets](<predictions.csv>) · [native-dtype raw branch arrays](<component_predictions_native.npz>) · [feature and parameter audits](<feature_audit.json>) · [branch metrics](<branch_metrics.csv>) · [timings](<timings.csv>) · [provenance](<provenance.json>) · [run log](<run.log>) · [Typst table](<comparison.typ>).

Seven new learner fits completed in {provenance['total_seconds']/60:.2f} minutes. Earlier full-refit control is reused, not retrained.
''')
        generate_typst(out, results)
        log(f'COMPLETE: {out / "summary.md"}')
        print(table, flush=True)
    except BaseException as exc:
        provenance.update({'status': 'failed', 'failure': repr(exc), 'traceback': traceback.format_exc()})
        dump(out / 'provenance.json', provenance)
        log(f'FAILED: {exc!r}')
        raise
    finally:
        log_file.close()


def generate_typst(out, results):
    indexed = results.set_index('variant')
    lines = ['// Generated from matched fulltest results; percentages, deltas in pp.',
             '#figure(', '  table(', '    columns: (3fr, 1fr, 1fr, 1fr),',
             '    align: (left, right, right, right),', '    inset: 7pt,', '    stroke: none,',
             '    table.hline(),', '    table.header([*Configuration*], [*MdAPE*], [*MAPE*], [*Δ MdAPE (pp)*]),',
             '    table.hline(),']
    for name in VARIANTS:
        row = indexed.loc[name]
        delta = '—' if name == 'full' else f'{row.delta_MdAPE_pp:+.4f}'
        label = LABELS[name] if name != 'base_only' else 'Only base features'
        lines.append(f'    [{label}], [{row.MdAPE:.4f}%], [{row.MAPE:.4f}%], [{delta}],')
    lines += ['    table.hline(),', '  ),', '  caption: [Cumulative ablations of OffMarketRefinedV6 on the matched fixed-origin fulltest (6,047 homes). Lower errors are better; positive Δ means removal hurts.],',
              ') <cumulative-ablation>', '', '#text(size: 9pt)[',
              '  All rows retain the same three learners and log-space blend weights (37.5%, 37.5%, 25%).',
              '  Only base features means the literal 44 columns returned by #raw("base_features()"),',
              '  including five text cues; CatBoost categorical/text declarations are adjusted to these inputs.',
              '  The second row retains semantic features; the third removes both semantic channels',
              '  but retains lexical text and spline features. Results use a development-selected window, not a fresh holdout.', ']']
    (out / 'comparison.typ').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
