"""Smooth additive offsets plus tree residuals and fair-value/list-price gaps."""
import numpy as np
import pandas as pd
from models.serialization import StableCatBoostRegressor as CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from models.BaseModel import BaseListAwareModel, BaseOffMarketModel
from models.catboost_v2 import _CatBoostFeatures
from models.features import SplineTensorBlock


class _HybridFeatures(_CatBoostFeatures):
    def __init__(self, iterations=1600):
        super().__init__(iterations=iterations)
        self.splines = SplineTensorBlock()

    def _features(self, homes, fit=False):
        frame = super()._features(homes, fit=fit)
        splines = self.splines.fit_transform(homes) if fit else self.splines.transform(homes)
        if self.LIST_AWARE:
            comps = self.comps.fit_transform(homes) if fit else self.comps.transform(homes)
            frame = pd.concat([frame, comps], axis=1).copy()
            frame['list_vs_comps'] = np.log(homes.list_price.to_numpy(float)) - frame.comp_log_price_est
            frame['list_ppsf'] = np.log(homes.list_price.to_numpy(float) / frame.area)
            frame['list_vs_building'] = frame['list_ppsf'] - frame['bldg_ppsf']
        return pd.concat([frame, splines], axis=1)


class ListAwareValueGapV5(_HybridFeatures, BaseListAwareModel):
    NAME = 'List-Aware Value Gap V5'
    MODEL_TAG = 'CatBoost'
    ENABLED = True
    LIST_AWARE = True

    def _train(self, data):
        frame = self._features(data, fit=True)
        target = np.log(data.sold_price.to_numpy(float) / data.list_price.to_numpy(float))
        self.model = CatBoostRegressor(
            iterations=self.iterations, depth=6, learning_rate=0.04,
            loss_function='RMSE', l2_leaf_reg=8, random_seed=43, thread_count=8,
            cat_features=['house_type_name', 'house_style', 'municipality', 'community', 'community_type'],
            text_features=['remarks_text'], feature_calcers=['BoW:top_tokens_count=2000'],
            allow_writing_files=False, verbose=False,
        )
        self.model.fit(frame, target)
        return self


class ListAwareValueGapHGBV5(ListAwareValueGapV5):
    NAME = 'List-Aware Value Gap HGB V5'
    MODEL_TAG = 'HGBR'

    def _numeric(self, frame, fit=False):
        frame = frame.drop(columns=['remarks_text', 'house_type_name', 'house_style', 'municipality', 'community_type']).copy()
        if fit:
            self.community_categories_ = sorted(frame.community.unique())
        values = frame.community.where(frame.community.isin(self.community_categories_))
        frame['community'] = pd.Categorical(values, categories=self.community_categories_)
        return frame

    def _train(self, data):
        frame = self._numeric(self._features(data, fit=True), fit=True)
        self.model = HistGradientBoostingRegressor(
            loss='absolute_error', learning_rate=0.04, max_iter=650,
            max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=5,
            early_stopping=False, categorical_features=['community'], random_state=43,
        )
        self.model.fit(frame, np.log(data.sold_price / data.list_price))
        return self

    def _predict(self, homes):
        if self.model is None:
            raise RuntimeError('Call train before predict')
        if homes.empty:
            return pd.Series(index=homes.index, dtype=float)
        target = self.model.predict(self._numeric(self._features(homes)))
        return pd.Series(homes.list_price.to_numpy(float) * np.exp(np.clip(target, -3, 3)), index=homes.index)
