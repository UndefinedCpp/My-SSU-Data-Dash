import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from models.BaseModel import BaseListAwareModel
from models.featuremodel_v3 import (
    ComparableBlock,
    ListAwareFeatureModelV2b,
    OffMarketFeatureModelV3,
    RemarksTermBlock,
    SplineTensorBlock,
)


def create_xgboost(
    objective: str = "reg:squarederror",
    *,
    n_estimators: int = 600,
    max_depth: int = 8,
    learning_rate: float = 0.04,
    min_child_weight: int = 5,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
) -> XGBRegressor:
    """Shared deterministic XGBoost configuration for the first experiments."""
    return XGBRegressor(
        objective=objective,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=8,
        n_jobs=8,
        random_state=0,
        verbosity=0,
        early_stopping=False,
    )


def _as_categories(
    frame: pd.DataFrame, categories: pd.Index | None = None
) -> tuple[pd.DataFrame, pd.Index]:
    values = frame["community"].astype(str)
    if categories is None:
        categories = pd.Index(sorted(values.unique()))
    frame = frame.copy()
    frame["community"] = pd.Categorical(values, categories=categories)
    return frame, categories


class OffMarketXGBoostV1(OffMarketFeatureModelV3):
    """
    Off-market XGBoost on the V3 features.
    """

    NAME = "Off-Market XGBoost V1"
    ENABLED = True
    MODEL_TAG = "XGB"

    def __init__(self):
        self.model = create_xgboost()
        self.comps = ComparableBlock()
        self.splines = SplineTensorBlock()
        self.text = RemarksTermBlock()
        self.community_categories_: pd.Index | None = None

    def _fit_features(self, data: pd.DataFrame) -> pd.DataFrame:
        frame, self.community_categories_ = _as_categories(
            super()._fit_features(data)
        )
        return frame

    def _transform_features(self, homes: pd.DataFrame) -> pd.DataFrame:
        if self.community_categories_ is None:
            raise RuntimeError("Call train before predict")
        frame, _ = _as_categories(
            super()._transform_features(homes), self.community_categories_
        )
        return frame


class ListAwareXGBoostV1(BaseListAwareModel):
    """
    List-aware XGBoost on the V2b features with a sold-to-list ratio target.
    """

    NAME = "List-Aware XGBoost V1"
    ENABLED = True
    MODEL_TAG = "XGB"

    def __init__(self):
        self.model = create_xgboost(
            objective="reg:absoluteerror",
            n_estimators=600,
            max_depth=6,
            learning_rate=0.04,
        )
        self.community_categories_: pd.Index | None = None

    def _train(self, data: pd.DataFrame) -> "ListAwareXGBoostV1":
        frame, self.community_categories_ = _as_categories(
            ListAwareFeatureModelV2b.make_features(data)
        )
        target = np.log(
            data["sold_price"].to_numpy(dtype=float)
            / data["list_price"].to_numpy(dtype=float)
        )
        self.model.fit(frame, target)
        return self

    def _predict(self, homes: pd.DataFrame) -> pd.Series:
        if self.community_categories_ is None:
            raise RuntimeError("Call train before predict")
        frame, _ = _as_categories(
            ListAwareFeatureModelV2b.make_features(homes),
            self.community_categories_,
        )
        ratio = np.exp(np.clip(self.model.predict(frame), -3, 3))
        return pd.Series(
            homes["list_price"].to_numpy(dtype=float) * ratio,
            index=homes.index,
        )
