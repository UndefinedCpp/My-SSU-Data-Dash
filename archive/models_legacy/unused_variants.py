"""Unused standalone variants, preserved outside active model discovery.

These implementations are reference code. Shared feature builders and learner
factories are imported from the active package; the submission path never
imports this module. For old checkpoints, use their original Git revision.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from models.BaseModel import BaseListAwareModel, BaseOffMarketModel
from models.attempt1 import _as_community_categories, _make_hgbr, _make_xgb
from models.features import (
    BuildingBlock, ComparableBlock, RemarksTermBlock, SplineTensorBlock,
    base_features,
)
from models.hybrid_v5 import _HybridFeatures
from models.serialization import StableCatBoostRegressor as CatBoostRegressor


class OffMarketBuildingV1(BaseOffMarketModel):
    """Off-market V3 + exact-building features with HGBR.

    Same feature matrix as the XGBoost variant; only the learner changes.
    """

    NAME = "Off-Market Building V1"
    ENABLED = True
    MODEL_TAG = "HGBR"

    def __init__(self):
        self.model = _make_hgbr(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=63,
            min_samples_leaf=40,
            l2_regularization=5.0,
        )
        self.comps = ComparableBlock()
        self.splines = SplineTensorBlock()
        self.text = RemarksTermBlock()
        self.building = BuildingBlock(include_ratio=False)

    def _fit_features(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(
            [
                base_features(data),
                self.comps.fit_transform(data),
                self.splines.fit_transform(data),
                self.text.fit_transform(data),
                self.building.fit_transform(data),
            ],
            axis=1,
        )

    def _transform_features(self, homes: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(
            [
                base_features(homes),
                self.comps.transform(homes),
                self.splines.transform(homes),
                self.text.transform(homes),
                self.building.transform(homes),
            ],
            axis=1,
        )

    def _train(self, data: pd.DataFrame) -> "OffMarketBuildingV1":
        X = self._fit_features(data)
        self.model.fit(X, np.log(data["sold_price"].to_numpy(dtype=float)))
        return self

    def _predict(self, homes: pd.DataFrame) -> pd.Series:
        X = self._transform_features(homes)
        raw = self.model.predict(X)
        return pd.Series(np.exp(np.clip(raw, 5, 20)), index=homes.index)


class ListAwareBuildingXGBoostV1(BaseListAwareModel):
    """List-aware ratio model with base features plus exact-building history."""

    NAME = "List-Aware Building XGBoost V1"
    ENABLED = True
    MODEL_TAG = "XGB"

    def __init__(self):
        self.model = _make_xgb(
            objective="reg:absoluteerror",
            n_estimators=600,
            max_depth=6,
            learning_rate=0.04,
        )
        self.building = BuildingBlock(include_ratio=True)
        self.community_categories_: pd.Index | None = None

    def _train(self, data: pd.DataFrame) -> "ListAwareBuildingXGBoostV1":
        X = pd.concat(
            [
                base_features(data, list_aware=True),
                self.building.fit_transform(data),
            ],
            axis=1,
        )
        X, self.community_categories_ = _as_community_categories(X)
        target = np.log(
            data["sold_price"].to_numpy(dtype=float)
            / data["list_price"].to_numpy(dtype=float)
        )
        self.model.fit(X, target)
        return self

    def _predict(self, homes: pd.DataFrame) -> pd.Series:
        if self.community_categories_ is None:
            raise RuntimeError("Call train before predict")
        X = pd.concat(
            [
                base_features(homes, list_aware=True),
                self.building.transform(homes),
            ],
            axis=1,
        )
        X, _ = _as_community_categories(X, self.community_categories_)
        ratio = np.exp(np.clip(self.model.predict(X), -3, 3))
        return pd.Series(
            homes["list_price"].to_numpy(dtype=float) * ratio,
            index=homes.index,
        )


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
