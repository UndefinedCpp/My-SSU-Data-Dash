"""Lightweight invariants for the isolated ablation runner (no model training)."""
import unittest

import numpy as np
import pandas as pd

from scripts.run_offmarket_ablations import (
    BUILDING_COLUMNS,
    COMP_COLUMNS,
    FeatureAudit,
    blend,
    metrics,
    paired_ci,
    predict,
)


class ToyFilter(FeatureAudit):
    pass


class TestAblationInvariants(unittest.TestCase):
    def frame(self):
        frame = pd.DataFrame({name: [1., 2., 3.] for name in
                              ['area', *COMP_COLUMNS, *BUILDING_COLUMNS,
                               'text:luxury', 'txt_0', 'semantic_0', 'historical_semantic_quality']})
        frame['community'] = pd.Categorical(['a', 'b', 'a'], categories=['a', 'b', 'c'])
        return frame

    def test_exact_block_removal_retains_order_dtypes_and_source(self):
        for prefix, columns in [('comp_', COMP_COLUMNS), ('bldg_', BUILDING_COLUMNS), ('', ())]:
            with self.subTest(prefix=prefix):
                source = self.frame()
                original = source.copy(deep=True)
                toy = ToyFilter(prefix)
                training = toy.filter_features(source, True)
                validation = toy.filter_features(source, False)
                expected = source.drop(columns=list(columns))
                pd.testing.assert_frame_equal(training, expected)
                pd.testing.assert_frame_equal(validation, expected)
                pd.testing.assert_frame_equal(source, original)
                self.assertEqual(toy.feature_audit['train']['removed_columns'], list(columns))

    def test_incomplete_block_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, 'Wrong removed columns'):
            ToyFilter('comp_').filter_features(self.frame().drop(columns=COMP_COLUMNS[0]), True)

    def test_schema_mismatch_is_rejected(self):
        toy = ToyFilter('bldg_')
        toy.filter_features(self.frame(), True)
        with self.assertRaisesRegex(AssertionError, 'schema differs'):
            toy.filter_features(self.frame().drop(columns='area'), False)

    def test_duplicate_features_are_rejected(self):
        with self.assertRaisesRegex(AssertionError, 'Duplicate feature'):
            ToyFilter('').filter_features(pd.DataFrame([[1., 2.]], columns=['area', 'area']), True)

    def test_nested_blend_and_final_rounding(self):
        parts = {'B': np.array([101.49, 201.51]), 'C': np.array([203.51, 301.49]),
                 'S': np.array([151.49, 251.51])}
        expected = np.exp(.375 * np.log(parts['B']) + .375 * np.log(parts['C']) + .25 * np.log(parts['S']))
        np.testing.assert_allclose(blend(parts), expected, atol=1e-10, rtol=0)
        self.assertFalse(np.array_equal(blend(parts), np.round(blend(parts))))

    def test_mixed_dtype_blend_matches_production_nesting(self):
        parts = {'B': np.array([101.49, 201.51], dtype=np.float32),
                 'C': np.array([203.51, 301.49], dtype=np.float64),
                 'S': np.array([151.49, 251.51], dtype=np.float32)}
        incumbent = np.exp(.5 * (np.log(parts['B']) + np.log(parts['C'])))
        production = np.exp((1 - .25) * np.log(incumbent) + .25 * np.log(parts['S']))
        np.testing.assert_array_equal(blend(parts), production)

    def test_misaligned_prediction_index_is_rejected(self):
        class ShuffledPredictor:
            def predict(self, homes):
                return pd.Series([1., 2.], index=[20, 10])
        with self.assertRaisesRegex(AssertionError, 'misalignment'):
            predict(ShuffledPredictor(), pd.DataFrame(index=[10, 20]))

    def test_prediction_preserves_native_float32_for_production_blend(self):
        class NativePredictor:
            def predict(self, homes):
                return pd.Series(np.array([101.49, 201.51], dtype=np.float32), index=homes.index)
        result = predict(NativePredictor(), pd.DataFrame(index=[10, 20]))
        self.assertEqual(result.dtype, np.dtype('float32'))

    def test_nonpositive_prediction_is_rejected(self):
        class BadPredictor:
            def predict(self, homes):
                return pd.Series([0., 2.], index=homes.index)
        with self.assertRaisesRegex(AssertionError, 'Invalid component'):
            predict(BadPredictor(), pd.DataFrame(index=[10, 20]))

    def test_paired_bootstrap_difference_of_medians(self):
        reference = np.array([1., 2., 8., 9.])
        variant = reference + 2.
        self.assertEqual(paired_ci(variant, reference, 100), (2., 2.))
        self.assertEqual(paired_ci(variant, reference, 100,
                                   groups=[np.array([0, 1]), np.array([2, 3])]), (2., 2.))
        self.assertEqual(paired_ci(reference, reference, 100), (0., 0.))

    def test_bootstrap_uses_difference_of_pooled_medians_not_median_pair_difference(self):
        variant = np.tile(np.array([0., 10., 11., 30., 50.]), 20)
        reference = np.tile(np.array([2., 3., 20., 21., 22.]), 20)
        rng = np.random.default_rng(42)
        pooled = []
        paired = []
        for _ in range(200):
            index = rng.integers(0, len(variant), len(variant))
            pooled.append(np.median(variant[index]) - np.median(reference[index]))
            paired.append(np.median(variant[index] - reference[index]))
        expected = np.quantile(pooled, [.025, .975])
        np.testing.assert_allclose(paired_ci(variant, reference, 200), expected)
        self.assertFalse(np.array_equal(expected, np.quantile(paired, [.025, .975])))

    def test_metrics_are_in_percentage_units(self):
        result = metrics(np.array([110., 180.]), np.array([100., 200.]))
        self.assertEqual(result['MdAPE'], 10.)
        self.assertEqual(result['MAPE'], 10.)
        self.assertEqual(result['median_signed_err'], 0.)


if __name__ == '__main__':
    unittest.main()
