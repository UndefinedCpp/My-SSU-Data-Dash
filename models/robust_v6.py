"""Absolute log-error tree learners with the richer listing-time feature set."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor

from models.BaseModel import BaseOffMarketModel
from models.hybrid_v5 import _HybridFeatures


class OffMarketRobustHGBV6(_HybridFeatures, BaseOffMarketModel):
    NAME = 'Off-Market Robust HGB V6'
    MODEL_TAG = 'HGBR'
    ENABLED = True

    def __init__(self, iterations=900):
        super().__init__(iterations=iterations)

    def _numeric(self, frame, fit=False):
        frame = frame.drop(columns=['remarks_text', 'house_type_name', 'house_style', 'municipality', 'community_type']).copy()
        if fit:
            self.community_categories_ = sorted(frame.community.unique())
        frame['community'] = pd.Categorical(frame.community.where(frame.community.isin(self.community_categories_)),
                                             categories=self.community_categories_)
        return frame

    def _train(self, data):
        frame = self._numeric(self._features(data, fit=True), fit=True)
        self.model = HistGradientBoostingRegressor(
            loss='absolute_error', learning_rate=0.04, max_iter=self.iterations,
            max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=5,
            early_stopping=False, categorical_features=['community'], random_state=44,
        )
        self.model.fit(frame, np.log(data.sold_price.to_numpy(float)))
        return self

    def _predict(self, homes):
        if self.model is None:
            raise RuntimeError('Call train before predict')
        if homes.empty:
            return pd.Series(index=homes.index, dtype=float)
        prediction = self.model.predict(self._numeric(self._features(homes)))
        return pd.Series(np.exp(np.clip(prediction, 5, 20)), index=homes.index)


class OffMarketRobustXGBV6(OffMarketRobustHGBV6):
    NAME = 'Off-Market Robust XGBoost V6'
    MODEL_TAG = 'XGB'

    def __init__(self, iterations=1600):
        super().__init__(iterations=iterations)

    def _train(self, data):
        frame = self._numeric(self._features(data, fit=True), fit=True)
        self.model = XGBRegressor(
            objective='reg:absoluteerror', n_estimators=self.iterations,
            max_depth=5, learning_rate=0.035, min_child_weight=30,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=8,
            tree_method='hist', enable_categorical=True, n_jobs=8,
            random_state=44, verbosity=0,
        )
        self.model.fit(frame, np.log(data.sold_price.to_numpy(float)))
        return self
