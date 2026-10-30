import numpy as np
import pandas as pd

from models.BaseModel import BaseListAwareModel
from models.featuremodel import square_feet
from models.featuremodel_v2b import ListAwareFeatureModelV2b
from models.featuremodel_v3 import OffMarketFeatureModelV3, create_hgbr
from models.xgboost_v1 import _as_categories, create_xgboost

try:  # optional dependency
    import xgboost as _xgboost  # noqa: F401
    _XGBOOST_AVAILABLE = True
except Exception:
    _XGBOOST_AVAILABLE = False

BUILDING_COLUMNS = (
    "bldg_n",
    "bldg_ppsf",
    "bldg_ppsf_std",
    "bldg_price",
    "bldg_area",
    "bldg_beds",
    "bldg_baths",
    "bldg_parking",
    "bldg_dom",
    "bldg_last_age_days",
    "bldg_n_12m",
    "bldg_ppsf_12m",
    "bldg_n_old",
    "bldg_ppsf_old",
    "bldg_ppsf_trend",
    "bldg_same_bed_n",
    "bldg_same_bed_ppsf",
    "bldg_same_type_n",
    "bldg_same_type_share",
    "bldg_area_dev",
    "bldg_beds_dev",
    "bldg_baths_dev",
    "bldg_parking_dev",
    "bldg_ppsf_gap_bed",
)

RATIO_COLUMNS = (
    "bldg_ratio",
    "bldg_ratio_std",
    "bldg_ratio_12m",
    "bldg_ratio_old",
    "bldg_ratio_trend",
    "bldg_same_bed_ratio",
    "bldg_ratio_gap_bed",
)


def _area(series: pd.Series) -> np.ndarray:
    return square_feet(series).to_numpy(dtype=float)


def _prepare_history(frame: pd.DataFrame, include_ratio: bool) -> pd.DataFrame:
    columns = [
        "date_listed", "date_sold", "sold_price", "house_type_name",
        "bedrooms", "washrooms", "parking_total", "house_area", "latitude",
        "longitude", "days_on_market",
    ]
    if include_ratio:
        columns.append("list_price")
    prepared = frame[columns].copy()
    prepared["_area"] = _area(prepared["house_area"])
    prepared["_log_ppsf"] = np.log(prepared["sold_price"] / prepared["_area"])
    prepared["_coord"] = list(zip(prepared["latitude"], prepared["longitude"]))
    prepared["_bed_group"] = prepared["bedrooms"].clip(1, 3)
    if include_ratio:
        prepared["_log_ratio"] = np.log(
            prepared["sold_price"] / prepared["list_price"]
        )
    return prepared


class BuildingBlock:
    """
    Historical features for the building at each query's exact coordinate.

    The dataset exhibit a clear structure of exact buildings (multiple columns
    have very similar coordinates), and it turns out that such features can be
    very informative. This is exactly what this component captures.
    """

    def __init__(self, include_ratio: bool = False):
        self.include_ratio = include_ratio
        self.history: pd.DataFrame | None = None
        self.columns = BUILDING_COLUMNS + (RATIO_COLUMNS if include_ratio else ())

    def _aggregate(
        self, history: pd.DataFrame, query: pd.DataFrame, cutoff: pd.Timestamp
    ) -> pd.DataFrame:
        out = pd.DataFrame(np.nan, index=query.index, columns=self.columns, dtype=float)
        if len(history) == 0 or len(query) == 0:
            return out

        h = history
        g = h.groupby("_coord", sort=False).agg(
            bldg_n=("_log_ppsf", "size"),
            bldg_ppsf=("_log_ppsf", "mean"),
            bldg_ppsf_std=("_log_ppsf", "std"),
            bldg_price=("sold_price", "mean"),
            bldg_area=("_area", "mean"),
            bldg_beds=("bedrooms", "mean"),
            bldg_baths=("washrooms", "mean"),
            bldg_parking=("parking_total", "mean"),
            bldg_dom=("days_on_market", "mean"),
            bldg_type_n=("house_type_name", "nunique"),
            bldg_last=("date_sold", "max"),
        )
        g["bldg_last_age_days"] = (cutoff - g["bldg_last"]).dt.days

        recent = h[h["date_sold"] >= cutoff - pd.Timedelta(days=365)]
        older = h[
            (h["date_sold"] >= cutoff - pd.Timedelta(days=730))
            & (h["date_sold"] < cutoff - pd.Timedelta(days=365))
        ]
        rg = recent.groupby("_coord", sort=False).agg(
            bldg_n_12m=("_log_ppsf", "size"),
            bldg_ppsf_12m=("_log_ppsf", "mean"),
        )
        og = older.groupby("_coord", sort=False).agg(
            bldg_n_old=("_log_ppsf", "size"),
            bldg_ppsf_old=("_log_ppsf", "mean"),
        )
        g = g.join(rg).join(og)
        g["bldg_ppsf_trend"] = g["bldg_ppsf_12m"] - g["bldg_ppsf_old"]

        sb = (
            h.groupby(["_coord", "_bed_group"], sort=False)
            .agg(
                bldg_same_bed_n=("_log_ppsf", "size"),
                bldg_same_bed_ppsf=("_log_ppsf", "mean"),
            )
            .reset_index()
        )
        st = (
            h.groupby(["_coord", "house_type_name"], sort=False)
            .size()
            .rename("bldg_same_type_n")
            .reset_index()
        )

        if self.include_ratio:
            ratio = h.groupby("_coord", sort=False).agg(
                bldg_ratio=("_log_ratio", "mean"),
                bldg_ratio_std=("_log_ratio", "std"),
            )
            r12 = recent.groupby("_coord", sort=False).agg(
                bldg_ratio_12m=("_log_ratio", "mean")
            )
            ro = older.groupby("_coord", sort=False).agg(
                bldg_ratio_old=("_log_ratio", "mean")
            )
            rsb = (
                h.groupby(["_coord", "_bed_group"], sort=False)
                .agg(bldg_same_bed_ratio=("_log_ratio", "mean"))
                .reset_index()
            )
            g = g.join(ratio).join(r12).join(ro)
            g["bldg_ratio_trend"] = g["bldg_ratio_12m"] - g["bldg_ratio_old"]

        q = query.copy()
        q["_coord"] = list(zip(q["latitude"], q["longitude"]))
        q["_bed_group"] = q["bedrooms"].clip(1, 3)
        merged = q.merge(g, left_on="_coord", right_index=True, how="left")
        merged = merged.merge(sb, on=["_coord", "_bed_group"], how="left")
        if self.include_ratio:
            merged = merged.merge(rsb, on=["_coord", "_bed_group"], how="left")
        merged = merged.merge(
            st, on=["_coord", "house_type_name"], how="left"
        )
        merged.index = query.index

        merged["bldg_same_type_share"] = (
            merged["bldg_same_type_n"] / merged["bldg_n"]
        )
        merged["bldg_area_dev"] = merged["_area"] - merged["bldg_area"]
        merged["bldg_beds_dev"] = merged["bedrooms"] - merged["bldg_beds"]
        merged["bldg_baths_dev"] = merged["washrooms"] - merged["bldg_baths"]
        merged["bldg_parking_dev"] = (
            merged["parking_total"] - merged["bldg_parking"]
        )
        merged["bldg_ppsf_gap_bed"] = (
            merged["bldg_ppsf"] - merged["bldg_same_bed_ppsf"]
        )
        if self.include_ratio:
            merged["bldg_ratio_gap_bed"] = (
                merged["bldg_ratio"] - merged["bldg_same_bed_ratio"]
            )
        return merged[list(self.columns)].astype(float)

    def _transform_by_month(
        self, homes: pd.DataFrame, history: pd.DataFrame
    ) -> pd.DataFrame:
        if len(homes) == 0:
            return pd.DataFrame(index=homes.index, columns=self.columns, dtype=float)
        frame = homes.copy()
        frame["_area"] = _area(frame["house_area"])
        listing_month = pd.to_datetime(frame["date_listed"]).dt.to_period("M")
        parts = []
        for period, group in frame.groupby(listing_month, sort=True):
            eligible = history[history["date_sold"] < period.start_time]
            parts.append(self._aggregate(eligible, group, period.start_time))
        if not parts:
            return pd.DataFrame(index=homes.index, columns=self.columns, dtype=float)
        return pd.concat(parts).reindex(homes.index)

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        self.history = _prepare_history(train, self.include_ratio)
        return self._transform_by_month(train, self.history)

    def transform(self, homes: pd.DataFrame) -> pd.DataFrame:
        if self.history is None:
            raise RuntimeError("Call fit_transform before transform")
        return self._transform_by_month(homes, self.history)


class OffMarketBuildingV1(OffMarketFeatureModelV3):

    NAME = "Off-Market Building V1"
    ENABLED = True
    MODEL_TAG = "HGBR"

    def __init__(self):
        super().__init__()
        self.building = BuildingBlock(include_ratio=False)

    def _fit_features(self, data: pd.DataFrame) -> pd.DataFrame:
        base = super()._fit_features(data)
        building = self.building.fit_transform(data)
        return pd.concat([base, building], axis=1)

    def _transform_features(self, homes: pd.DataFrame) -> pd.DataFrame:
        base = super()._transform_features(homes)
        building = self.building.transform(homes)
        return pd.concat([base, building], axis=1)


class OffMarketBuildingXGBoostV1(OffMarketBuildingV1):

    NAME = "Off-Market Building XGBoost V1"
    ENABLED = _XGBOOST_AVAILABLE
    MODEL_TAG = "XGB"

    def __init__(self):
        super().__init__()
        
        self._as_categories = _as_categories
        self.model = create_xgboost()
        self.community_categories_: pd.Index | None = None

    def _fit_features(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = super()._fit_features(data)
        frame, self.community_categories_ = self._as_categories(frame)
        return frame

    def _transform_features(self, homes: pd.DataFrame) -> pd.DataFrame:
        if self.community_categories_ is None:
            raise RuntimeError("Call train before predict")
        frame = super()._transform_features(homes)
        frame, _ = self._as_categories(frame, self.community_categories_)
        return frame


class ListAwareBuildingV1(BaseListAwareModel):

    NAME = "List-Aware Building V1"
    ENABLED = True
    MODEL_TAG = "HGBR"

    def __init__(self):
        self.model = create_hgbr(
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
                ListAwareFeatureModelV2b.make_features(data),
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
                ListAwareFeatureModelV2b.make_features(homes),
                self.building.transform(homes),
            ],
            axis=1,
        )
        ratio = np.exp(np.clip(self.model.predict(X), -3, 3))
        return pd.Series(
            homes["list_price"].to_numpy(dtype=float) * ratio,
            index=homes.index,
        )


class ListAwareBuildingXGBoostV1(ListAwareBuildingV1):

    NAME = "List-Aware Building XGBoost V1"
    ENABLED = _XGBOOST_AVAILABLE
    MODEL_TAG = "XGB"

    def __init__(self):
        super().__init__()
        self._as_categories = _as_categories
        self.model = create_xgboost(
            objective="reg:absoluteerror",
            n_estimators=600,
            max_depth=6,
            learning_rate=0.04,
        )
        self.community_categories_: pd.Index | None = None

    def _train(self, data: pd.DataFrame) -> "ListAwareBuildingXGBoostV1":
        X = pd.concat(
            [
                ListAwareFeatureModelV2b.make_features(data),
                self.building.fit_transform(data),
            ],
            axis=1,
        )
        X, self.community_categories_ = self._as_categories(X)
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
                ListAwareFeatureModelV2b.make_features(homes),
                self.building.transform(homes),
            ],
            axis=1,
        )
        X, _ = self._as_categories(X, self.community_categories_)
        ratio = np.exp(np.clip(self.model.predict(X), -3, 3))
        return pd.Series(
            homes["list_price"].to_numpy(dtype=float) * ratio,
            index=homes.index,
        )
