"""Final self-contained models: off-market building XGBoost and list-aware HGBR."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from models.BaseModel import BaseListAwareModel, BaseOffMarketModel
from models.features import (
    BuildingBlock,
    ComparableBlock,
    RemarksTermBlock,
    SplineTensorBlock,
    base_features,
)

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except Exception:  # optional dependency / missing libomp
    XGBRegressor = None
    XGBOOST_AVAILABLE = False


def _make_hgbr(**overrides) -> HistGradientBoostingRegressor:
    params = dict(
        categorical_features=["community"],
        early_stopping=False,
        random_state=0,
    )
    params.update(overrides)
    return HistGradientBoostingRegressor(**params)


def _make_xgb(**overrides) -> "XGBRegressor":
    params = dict(
        objective="reg:squarederror",
        n_estimators=600,
        learning_rate=0.04,
        max_depth=8,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=8,
        n_jobs=8,
        random_state=0,
        verbosity=0,
        early_stopping=False,
    )
    params.update(overrides)
    return XGBRegressor(**params)


def _as_community_categories(
    frame: pd.DataFrame, categories: pd.Index | None = None
) -> tuple[pd.DataFrame, pd.Index]:
    values = frame["community"].astype(str)
    if categories is None:
        categories = pd.Index(sorted(values.unique()))
    frame = frame.copy()
    frame["community"] = pd.Categorical(values, categories=categories)
    return frame, categories


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


class OffMarketBuildingXGBoostV1(BaseOffMarketModel):
    """Off-market V3 features plus exact-building history, XGBoost learner.

    Feature order: base property/geography, comparable sales, space-time
    splines, remarks terms, then building history.
    """

    NAME = "Off-Market Building XGBoost V1"
    ENABLED = XGBOOST_AVAILABLE
    MODEL_TAG = "XGB"

    def __init__(self):
        self.model = _make_xgb()
        self.comps = ComparableBlock()
        self.splines = SplineTensorBlock()
        self.text = RemarksTermBlock()
        self.building = BuildingBlock(include_ratio=False)
        self.community_categories_: pd.Index | None = None

    def _fit_features(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = pd.concat(
            [
                base_features(data),
                self.comps.fit_transform(data),
                self.splines.fit_transform(data),
                self.text.fit_transform(data),
                self.building.fit_transform(data),
            ],
            axis=1,
        )
        frame, self.community_categories_ = _as_community_categories(frame)
        return frame

    def _transform_features(self, homes: pd.DataFrame) -> pd.DataFrame:
        if self.community_categories_ is None:
            raise RuntimeError("Call train before predict")
        frame = pd.concat(
            [
                base_features(homes),
                self.comps.transform(homes),
                self.splines.transform(homes),
                self.text.transform(homes),
                self.building.transform(homes),
            ],
            axis=1,
        )
        frame, _ = _as_community_categories(frame, self.community_categories_)
        return frame

    def _train(self, data: pd.DataFrame) -> "OffMarketBuildingXGBoostV1":
        X = self._fit_features(data)
        self.model.fit(X, np.log(data["sold_price"].to_numpy(dtype=float)))
        return self

    def _predict(self, homes: pd.DataFrame) -> pd.Series:
        X = self._transform_features(homes)
        raw = self.model.predict(X)
        return pd.Series(np.exp(np.clip(raw, 5, 20)), index=homes.index)


class ListAwareBuildingV1(BaseListAwareModel):
    """List-aware ratio model with base features plus exact-building history."""

    NAME = "List-Aware Building V1"
    ENABLED = True
    MODEL_TAG = "HGBR"

    def __init__(self):
        self.model = _make_hgbr(
            loss="absolute_error",
            max_iter=300,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
        )
        self.building = BuildingBlock(include_ratio=True)

    def _train(self, data: pd.DataFrame) -> "ListAwareBuildingV1":
        X = pd.concat(
            [
                base_features(data, list_aware=True),
                self.building.fit_transform(data),
            ],
            axis=1,
        )
        target = np.log(
            data["sold_price"].to_numpy(dtype=float)
            / data["list_price"].to_numpy(dtype=float)
        )
        self.model.fit(X, target)
        return self

    def _predict(self, homes: pd.DataFrame) -> pd.Series:
        X = pd.concat(
            [
                base_features(homes, list_aware=True),
                self.building.transform(homes),
            ],
            axis=1,
        )
        ratio = np.exp(np.clip(self.model.predict(X), -3, 3))
        return pd.Series(
            homes["list_price"].to_numpy(dtype=float) * ratio,
            index=homes.index,
        )

class ListAwareBuildingXGBoostV1(BaseListAwareModel):
    """List-aware ratio model with base features plus exact-building history."""

    NAME = "List-Aware Building XGBoost V1"
    ENABLED = XGBOOST_AVAILABLE
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

