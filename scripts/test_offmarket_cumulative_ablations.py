"""No-training regression tests for cumulative joint/base-only ablations."""
import unittest

import numpy as np
import pandas as pd

from scripts.run_offmarket_cumulative_ablations import (
    Audit, BaseBuilding, BaseCatBoost, BaseRobust, JOINT_COLUMNS,
    adapted_base_cat_params, base_features,
)


def homes():
    return pd.DataFrame({
        'house_area': ['1000-1200', '1500-2000', '800-1000'],
        'bedrooms': [2., 3., 1.], 'washrooms': [1., 2., 1.],
        'parking_total': [1., 2., 0.], 'latitude': [43.7, 43.8, 43.6],
        'longitude': [-79.4, -79.3, -79.5],
        'date_listed': pd.to_datetime(['2025-01-01', '2025-02-01', '2025-03-01']),
        'house_type_name': ['Condo Apt', 'Detached', 'Condo Apt'],
        'house_style': ['Apartment', '2-Storey', 'Apartment'],
        'municipality': ['Toronto', 'North York', 'Toronto'],
        'community': ['a', 'b', 'a'], 'lot_front': [0., 30., 0.],
        'lot_depth': [0., 100., 0.], 'remarks': ['Renovated luxury view', 'Original potential', ''],
    }, index=[10, 20, 30])


class TestCumulativeAblations(unittest.TestCase):
    def test_literal_base_schema_and_text_cues(self):
        values = base_features(homes())
        self.assertEqual(len(values.columns), 44)
        self.assertEqual(sum(c.startswith('text:') for c in values), 5)
        self.assertTrue(values['text:renovated'].iloc[0] > 0)
        self.assertFalse(any(c.startswith(('txt_', 'spl_', 'tensor_', 'comp_', 'bldg_', 'semantic_')) for c in values))

    def test_base_building_keeps_exact_values_and_unknown_categories(self):
        model = BaseBuilding()
        frame = homes()
        result = model._fit_features(frame)
        expected = base_features(frame)
        expected.community = pd.Categorical(expected.community, categories=['a', 'b'])
        pd.testing.assert_frame_equal(result, expected)
        query = frame.copy()
        query.loc[10, 'community'] = 'unknown'
        transformed = model._transform_features(query)
        self.assertTrue(pd.isna(transformed.loc[10, 'community']))
        self.assertEqual(list(transformed), list(expected))

    def test_base_robust_has_no_stale_raw_categorical_drops(self):
        model = BaseRobust()
        numeric = model._numeric(model._features(homes(), fit=True), fit=True)
        self.assertEqual(list(numeric), list(base_features(homes())))
        self.assertIsInstance(numeric.community.dtype, pd.CategoricalDtype)
        query = homes()
        query.loc[10, 'community'] = 'unknown'
        output = model._numeric(model._features(query))
        self.assertTrue(pd.isna(output.loc[10, 'community']))

    def test_required_base_catboost_parameter_changes_only(self):
        params = {'iterations': 1200, 'depth': 7, 'loss_function': 'RMSE', 'random_seed': 42,
                  'cat_features': ['house_type_name', 'community'], 'text_features': ['remarks_text'],
                  'feature_calcers': ['BoW:top_tokens_count=1500']}
        output = adapted_base_cat_params(params)
        self.assertEqual(output, {'iterations': 1200, 'depth': 7, 'loss_function': 'RMSE',
                                  'random_seed': 42, 'cat_features': ['community']})
        self.assertIn('text_features', params)
        model = BaseCatBoost(output)
        pd.testing.assert_frame_equal(model._features(homes(), fit=True), base_features(homes()))

    def test_joint_removal_exact_38_columns_preserves_remaining_categories(self):
        frame = pd.DataFrame({c: [1., 2.] for c in ['area', *sorted(JOINT_COLUMNS), 'text:luxury']})
        frame['community'] = pd.Categorical(['a', 'b'], categories=['a', 'b', 'c'])
        source = frame.copy()
        model = Audit()
        expected = frame.drop(columns=list(JOINT_COLUMNS))
        pd.testing.assert_frame_equal(model.remove_history(frame, True), expected)
        pd.testing.assert_frame_equal(model.remove_history(frame, False), expected)
        pd.testing.assert_frame_equal(frame, source)
        self.assertEqual(len(model.feature_audit['train']['removed_columns']), 38)

    def test_joint_removal_rejects_partial_block(self):
        with self.assertRaises(KeyError):
            Audit().remove_history(pd.DataFrame({'area': [1.]}), True)

    def test_numeric_missing_preprocessing_adds_no_columns(self):
        query = homes()
        query.loc[10, 'bedrooms'] = np.nan
        query.loc[20, 'latitude'] = np.inf
        model = BaseRobust()
        result = model._features(query, fit=True)
        self.assertEqual(list(result), list(base_features(query)))
        self.assertEqual(result.loc[10, 'bedrooms'], -999)
        self.assertEqual(result.loc[20, 'latitude'], -999)
        self.assertTrue(pd.isna(BaseBuilding()._fit_features(query).loc[10, 'bedrooms']))


if __name__ == '__main__':
    unittest.main()
