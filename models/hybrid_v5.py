"""Shared value-gap features and HGB base for the historical text stack."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from models.BaseModel import BaseListAwareModel
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


class ListAwareValueGapHGBV5(_HybridFeatures, BaseListAwareModel):
    """Implementation base for the selected historical-text model."""

    NAME = 'List-Aware Value Gap HGB V5'
    MODEL_TAG = 'HGBR'
    ENABLED = False
    LIST_AWARE = True

    # Learner settings. Subclasses may override these without copying `_train`.
    MAX_ITER = 650
    LEARNING_RATE = 0.04
    MAX_LEAF_NODES = 31
    MIN_SAMPLES_LEAF = 40
    L2_REGULARIZATION = 5
    RANDOM_STATE = 43

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
            loss='absolute_error', learning_rate=self.LEARNING_RATE,
            max_iter=self.MAX_ITER, max_leaf_nodes=self.MAX_LEAF_NODES,
            min_samples_leaf=self.MIN_SAMPLES_LEAF,
            l2_regularization=self.L2_REGULARIZATION,
            early_stopping=False, categorical_features=['community'],
            random_state=self.RANDOM_STATE,
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
