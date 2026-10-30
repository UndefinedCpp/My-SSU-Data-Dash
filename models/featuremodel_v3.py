from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import SplineTransformer

from models.BaseModel import BaseListAwareModel, BaseOffMarketModel
from models.featuremodel import square_feet
from models.featuremodel_v2b import ListAwareFeatureModelV2b, OffMarketFeatureModelV2b

LAT_CENTER = 43.7

# Text terms that vectorizer will train on
TEXT_TERMS = (
    "as is", "fully", "first", "downsizers", "young professionals", "investors",
    "starting", "discerning", "special", "family", "growing", "offered",
    "renovated", "move up", "potential", "first time", "vision", "entertaining",
    "redone", "entertain", "move", "time", "new", "original", "older", "custom",
    "growing family", "couple", "opportunity", "central air", "grow", "front",
    "discerning buyer", "turnkey", "investor", "project", "starting out", "unit",
    "cozy", "spa like", "wide plank", "top", "room to", "beautifully", "needs",
    "bones", "overhaul",
)

COMP_COLUMNS = (
    "comp_log_price_est",
    "comp_log_ppsf",
    "comp_local_ppsf_median",
    "comp_price_std",
    "comp_ppsf_std",
    "comp_count",
    "comp_effective_n",
    "comp_nearest_km",
    "comp_mean_distance_km",
    "comp_mean_age_days",
    "comp_same_type_share",
    "comp_area_similarity",
    "comp_recent_180d_share",
    "comp_momentum",
)

warnings.filterwarnings("ignore", message="All-NaN slice encountered")


def _area(series: pd.Series) -> np.ndarray:
    return square_feet(series).to_numpy(dtype=float)


def _day(series: pd.Series) -> np.ndarray:
    return pd.to_datetime(series).to_numpy(dtype="datetime64[D]").astype("int64")


def _project_coordinates(latitude, longitude) -> np.ndarray:
    """
    Project latitude and longitude to approximate x and y coordinates in kilometres.
    """
    x = np.asarray(longitude, dtype=float) * 111.32 * np.cos(np.deg2rad(LAT_CENTER))
    y = np.asarray(latitude, dtype=float) * 111.32
    return np.column_stack([x, y])


def _prepare_history(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["_area_sqft"] = _area(prepared["house_area"])
    prepared["_sold_day"] = _day(prepared["date_sold"])
    return prepared


def _prepare_query(frame: pd.DataFrame) -> pd.DataFrame:
    if "_area_sqft" in frame.columns and "_listed_day" in frame.columns:
        return frame
    prepared = frame.copy()
    prepared["_area_sqft"] = _area(prepared["house_area"])
    prepared["_listed_day"] = _day(prepared["date_listed"])
    return prepared


def create_hgbr(**overrides) -> HistGradientBoostingRegressor:
    params = dict(
        categorical_features=["community"],
        early_stopping=False,
        random_state=0,
    )
    params.update(overrides)
    return HistGradientBoostingRegressor(**params)


def compute_comparables(
    history: pd.DataFrame,
    query: pd.DataFrame,
    *,
    candidates: int = 48,
    neighbors: int = 48,
    radius_km: float = 4.0,
    half_life_days: float = 180.0,
    max_age_days: float = 730.0,
    area_bandwidth: float = 0.45,
    max_distance_km: float = 8.0,
) -> pd.DataFrame:
    """
    Build 14 local comparable-sale features for each query home.

    For every query we take the nearest `candidates` earlier sales in a
    projected KD-tree, reject sales that are too old or too far away, and
    calculate a "similarity weight".

    The kept neighbours are ranked by that specific weight. The returned
    columns describe the weighted local price level, dispersion, comparable
    count, effective sample size, distance/age profile, same-type share,
    area similarity, and recent price momentum.

    KD-Tree is an efficient data structure for nearest-neighbour searches.
    """
    # Initialize the output DataFrame with NaNs for all query homes.
    out = pd.DataFrame(np.nan, index=query.index, columns=COMP_COLUMNS, dtype=float)
    if len(query) == 0 or len(history) == 0:
        return out

    h = history
    h_lat = h["latitude"].to_numpy(float)
    h_lon = h["longitude"].to_numpy(float)
    h_area = (
        h["_area_sqft"].to_numpy(float) if "_area_sqft" in h.columns else _area(h["house_area"])
    )
    h_price = h["sold_price"].to_numpy(float)
    h_type = h["house_type_name"].astype(str).to_numpy()
    h_sold = (
        h["_sold_day"].to_numpy("int64") if "_sold_day" in h.columns
        else pd.to_datetime(h["date_sold"]).to_numpy(dtype="datetime64[D]").astype("int64")
    )
    valid = (
        np.isfinite(h_lat) & np.isfinite(h_lon) & np.isfinite(h_area) & (h_area > 0)
        & np.isfinite(h_price) & (h_price > 0)
    )
    if valid.sum() < 10:
        return out

    # Construct projected KD Tree
    h_lat, h_lon = h_lat[valid], h_lon[valid]
    h_area, h_price = h_area[valid], h_price[valid]
    h_type, h_sold = h_type[valid], h_sold[valid]

    history_coordinates = _project_coordinates(h_lat, h_lon)
    tree = cKDTree(history_coordinates)
    h_log_price = np.log(h_price)
    h_log_area = np.log(h_area)
    h_log_ppsf = h_log_price - h_log_area
    h_type_code, type_levels = pd.factorize(h_type)

    # Extract query home features and prepare for KD-tree search.
    q_lat = query["latitude"].to_numpy(float)
    q_lon = query["longitude"].to_numpy(float)
    q_area = (
        query["_area_sqft"].to_numpy(float) if "_area_sqft" in query.columns
        else _area(query["house_area"])
    )
    q_type = query["house_type_name"].astype(str).to_numpy()
    q_listed = (
        query["_listed_day"].to_numpy("int64") if "_listed_day" in query.columns
        else pd.to_datetime(query["date_listed"]).to_numpy(dtype="datetime64[D]").astype("int64")
    )
    q_log_area = np.log(np.where(q_area > 0, q_area, np.nan))
    q_type_code = pd.Categorical(q_type, categories=type_levels).codes

    # Filter out query homes with invalid features
    finite = np.isfinite(q_lat) & np.isfinite(q_lon) & np.isfinite(q_area) & (q_area > 0)
    positions = np.flatnonzero(finite)
    if len(positions) == 0:
        return out

    # KD-Tree search
    nn = min(candidates, len(h_area))
    for start in range(0, len(positions), 10_000):
        pos = positions[start : start + 10_000]
        distance, index = tree.query(
            _project_coordinates(q_lat[pos], q_lon[pos]), k=nn
        )
        if nn == 1:
            distance, index = distance[:, None], index[:, None]

        distance_km = distance
        age_days = q_listed[pos, None] - h_sold[index]
        same_type = h_type_code[index] == q_type_code[pos, None]
        area_gap = np.abs(h_log_area[index] - q_log_area[pos, None])

        keep = (
            (age_days >= 0)
            & (age_days <= max_age_days)
            & (distance_km <= max_distance_km)
        )
        weights = (
            np.exp(-distance_km / radius_km)
            * np.exp(-np.log(2) * age_days / half_life_days)
            * np.exp(-area_gap / area_bandwidth)
            * np.where(same_type, 1.0, 0.25)
        )
        weights = np.where(keep, weights, 0.0)

        # Rank the candidate pool by the combined similarity weight, not by
        # raw distance alone. This avoids arbitrary tie-breaking in dense
        # condo towers, where hundreds of sales share one coordinate.
        if neighbors < nn:
            top = np.argsort(-weights, axis=1)[:, :neighbors]
            selected = np.zeros_like(weights, dtype=bool)
            np.put_along_axis(selected, top, True, axis=1)
            selected &= weights > 0
            weights = np.where(selected, weights, 0.0)
        else:
            selected = weights > 0

        sum_w = weights.sum(axis=1)
        sum_w2 = (weights**2).sum(axis=1)
        count = (weights > 0).sum(axis=1)
        usable = sum_w > 1e-12
        inv_w = np.divide(1.0, sum_w, out=np.zeros_like(sum_w), where=usable)

        ppsf = h_log_ppsf[index]    # Price Per Square Foot
        log_price = h_log_price[index]
        mean_ppsf = (weights * ppsf).sum(axis=1) * inv_w
        mean_price = (weights * log_price).sum(axis=1) * inv_w
        std_price = np.sqrt(
            np.maximum((weights * (log_price - mean_price[:, None]) ** 2).sum(axis=1) * inv_w, 0)
        )
        std_ppsf = np.sqrt(
            np.maximum((weights * (ppsf - mean_ppsf[:, None]) ** 2).sum(axis=1) * inv_w, 0)
        )

        recent = age_days < 180
        recent_w = (weights * recent).sum(axis=1)
        older_w = sum_w - recent_w
        recent_ppsf = np.divide(
            (weights * recent * ppsf).sum(axis=1), recent_w,
            out=np.full_like(sum_w, np.nan), where=recent_w > 1e-12,
        )
        older_ppsf = np.divide(
            (weights * ~recent * ppsf).sum(axis=1), older_w,
            out=np.full_like(sum_w, np.nan), where=older_w > 1e-12,
        )

        # Construct the final feature block for each query home
        block = np.column_stack([
            mean_ppsf + q_log_area[pos], # size-adjusted local value
            mean_ppsf, # local log price per sq ft
            np.nanmedian(np.where(selected, ppsf, np.nan), axis=1),  # unweighted local median
            std_price, # standard deviation of log price
            std_ppsf, # standard deviation of log price per sq ft
            count, # number of comparable sales
            np.divide(sum_w**2, sum_w2, out=np.zeros_like(sum_w), where=sum_w2 > 0), # effective sample size
            np.where(count > 0, np.where(selected, distance_km, np.inf).min(axis=1), np.nan), # minimum distance
            (weights * distance_km).sum(axis=1) * inv_w, # weighted average distance
            (weights * age_days).sum(axis=1) * inv_w, # weighted average age
            (weights * same_type).sum(axis=1) * inv_w, # weighted same-type share
            (weights * np.exp(-area_gap)).sum(axis=1) * inv_w, # weighted area similarity
            recent_w * inv_w, # recent weight fraction
            recent_ppsf - older_ppsf, # recent price momentum
        ])
        block[~usable, :] = np.nan
        out.iloc[pos, :] = block
    return out


class ComparableBlock:
    """
    Fit and apply `compute_comparables` with per-listing-month cutoffs.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.history: pd.DataFrame | None = None

    @staticmethod
    def _eligible_history(history: pd.DataFrame, period: pd.Period) -> pd.DataFrame:
        return history[history["date_sold"] < period.start_time]

    def _transform_by_month(
        self, homes: pd.DataFrame, history: pd.DataFrame
    ) -> pd.DataFrame:
        homes = _prepare_query(homes)
        listing_month = homes["date_listed"].dt.to_period("M")
        parts = []
        for period, group in homes.groupby(listing_month, sort=True):
            eligible = self._eligible_history(history, period)
            if len(eligible) < 50:
                continue
            parts.append(compute_comparables(eligible, group, **self.kwargs))
        if not parts:
            return pd.DataFrame(np.nan, index=homes.index, columns=COMP_COLUMNS)
        return pd.concat(parts).reindex(homes.index)

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        columns = [
            "latitude", "longitude", "house_area", "house_type_name",
            "date_listed", "date_sold", "sold_price",
        ]
        self.history = _prepare_history(train[columns])
        return self._transform_by_month(train, self.history)

    def transform(self, homes: pd.DataFrame) -> pd.DataFrame:
        if self.history is None:
            raise RuntimeError("Call fit_transform before transform")
        return self._transform_by_month(homes, self.history)


class SplineTensorBlock:
    """
    Cubic spline bases for latitude, longitude and listing month.
    """

    def __init__(self, n_knots: int = 5):
        self.n_knots = n_knots
        self.splines: list[SplineTransformer] = []
        self.fill: np.ndarray | None = None

    @staticmethod
    def _coordinates(homes: pd.DataFrame) -> np.ndarray:
        dates = pd.to_datetime(homes["date_listed"], errors="coerce")
        month_index = dates.dt.to_period("M").astype("int64").to_numpy(dtype=float)
        return np.column_stack([
            pd.to_numeric(homes["latitude"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(homes["longitude"], errors="coerce").to_numpy(dtype=float),
            month_index,
        ])

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        coordinates = self._coordinates(train)
        self.fill = np.nanmedian(
            np.where(np.isfinite(coordinates), coordinates, np.nan), axis=0
        )
        self.fill = np.where(np.isfinite(self.fill), self.fill, 0.0)
        coordinates = np.where(np.isfinite(coordinates), coordinates, self.fill)
        self.splines = [
            SplineTransformer(
                n_knots=self.n_knots,
                degree=3,
                include_bias=False,
                extrapolation="constant", # TODO
            ).fit(coordinates[:, [i]])
            for i in range(3)
        ]
        return self.transform(train)

    def transform(self, homes: pd.DataFrame) -> pd.DataFrame:
        if not self.splines or self.fill is None:
            raise RuntimeError("Call fit_transform before transform")
        coordinates = self._coordinates(homes)
        coordinates = np.where(np.isfinite(coordinates), coordinates, self.fill)
        pieces = [
            spline.transform(coordinates[:, [i]])
            for i, spline in enumerate(self.splines)
        ]
        data = {
            f"spl_{axis}_{j}": part[:, j]
            for axis, part in enumerate(pieces)
            for j in range(part.shape[1])
        }
        for a, b, label in ((0, 1, "space"), (0, 2, "lat_time"), (1, 2, "lon_time")):
            cross = np.einsum("ij,ik->ijk", pieces[a], pieces[b])
            for j in range(cross.shape[1]):
                for k in range(cross.shape[2]):
                    data[f"tensor_{label}_{j}_{k}"] = cross[:, j, k]
        return pd.DataFrame(data, index=homes.index, dtype="float32")


class RemarksTermBlock:
    """
    Binary encoding of a fixed 47-term vocabulary in `remarks`.
    """

    _cache: dict[str, np.ndarray] = {}
    _vectorizer: CountVectorizer | None = None

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()
        cls._vectorizer = None

    def _frame(self, homes: pd.DataFrame) -> pd.DataFrame:
        ids = homes["ml_num"].astype(str).to_numpy()
        remarks = homes["remarks"].fillna("").astype(str).to_numpy()
        missing = [i for i in ids if i not in self._cache]
        if missing:
            text_by_id = dict(zip(ids, remarks))
            documents = [text_by_id[i] for i in missing]
            if self._vectorizer is None:
                self._vectorizer = CountVectorizer(
                    vocabulary=list(TEXT_TERMS),
                    ngram_range=(1, 2),
                    binary=True,
                    dtype=np.float32,
                )
                matrix = self._vectorizer.fit_transform(documents)
            else:
                matrix = self._vectorizer.transform(documents)
            matrix = matrix.toarray()
            for row, identifier in enumerate(missing):
                self._cache[identifier] = matrix[row]
        values = np.vstack([self._cache[i] for i in ids])
        return pd.DataFrame(
            values,
            index=homes.index,
            columns=[f"txt_{i}" for i in range(values.shape[1])],
            dtype="float32",
        )

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self._frame(train)

    def transform(self, homes: pd.DataFrame) -> pd.DataFrame:
        if self._vectorizer is None and not self._cache:
            raise RuntimeError("Call fit_transform before transform")
        return self._frame(homes)



class OffMarketFeatureModelV3(BaseOffMarketModel):
    """
    Off-market HGBR model with even more advanced features.
    
    1. V2b features: area, rooms, parking, lot, coordinates, distance to
       centre, style/municipality flags, community category, month of year,
       remarks cue counts.
    2. Comparable sales: 14 features from nearby earlier sales, weighted by
       distance (4 km bandwidth), recency (180-day half-life), area similarity,
       and same-property-type bonus.
    3. Space-time splines: 126 features from cubic spline bases for latitude,
       longitude and listing month, including latitude x longitude,
       latitude x time, and longitude x time tensor interactions.
    4. Remarks terms: 47 binary term-presence features.

    Besides the feature upgrades, the model also benefits from more iterations
    and careful regularization. This makes the training time significantly
    longer but yields more promising results.
    """

    NAME = "Off-Market V3"
    ENABLED = True
    MODEL_TAG = "HGBR"

    def __init__(self):
        self.model = create_hgbr(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=63,
            min_samples_leaf=40,
            l2_regularization=5.0,
        )
        self.comps = ComparableBlock()
        self.splines = SplineTensorBlock()
        self.text = RemarksTermBlock()

    def _fit_features(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(
            [
                OffMarketFeatureModelV2b.make_features(data),
                self.comps.fit_transform(data),
                self.splines.fit_transform(data),
                self.text.fit_transform(data),
            ],
            axis=1,
        )

    def _transform_features(self, homes: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(
            [
                OffMarketFeatureModelV2b.make_features(homes),
                self.comps.transform(homes),
                self.splines.transform(homes),
                self.text.transform(homes),
            ],
            axis=1,
        )

    def _train(self, data: pd.DataFrame) -> "OffMarketFeatureModelV3":
        X = self._fit_features(data)
        self.model.fit(X, np.log(data["sold_price"].to_numpy(dtype=float)))
        return self

    def _predict(self, homes: pd.DataFrame) -> pd.Series:
        X = self._transform_features(homes)
        raw = self.model.predict(X)
        return pd.Series(np.exp(np.clip(raw, 5, 20)), index=homes.index)



class ListAwareFeatureModelV3(BaseListAwareModel):
    """
    List-aware HGBR model trained on the sold-to-list ratio.

    Training target switched to `log(target/list_price)`.

    Unlike the off-market V3 model, this model does not use advanced features.
    This is intentional because it appears that the listed price already carries
    much overall signal and adding more features make the model less robust.
    Instead, simple improvements are already good enough.
    """

    NAME = "List-aware V3"
    ENABLED = True
    MODEL_TAG = "HGBR"

    def __init__(self):
        self.model = create_hgbr(
            loss="absolute_error", # useful
            max_iter=300,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
        )

    def _train(self, data: pd.DataFrame) -> "ListAwareFeatureModelV3":
        X = ListAwareFeatureModelV2b.make_features(data)
        target = np.log(
            data["sold_price"].to_numpy(dtype=float)
            / data["list_price"].to_numpy(dtype=float)
        )
        self.model.fit(X, target)
        return self

    def _predict(self, homes: pd.DataFrame) -> pd.Series:
        X = ListAwareFeatureModelV2b.make_features(homes)
        ratio = np.exp(self.model.predict(X))
        return pd.Series(
            homes["list_price"].to_numpy(dtype=float) * ratio,
            index=homes.index,
        )


# ---------------------------------------------------------------------------
# Spatial residual correction
# ---------------------------------------------------------------------------

RESIDUAL_MAX_DISTANCE_KM = 1.5
RESIDUAL_MAX_NEIGHBORS = 50
RESIDUAL_MIN_NEIGHBORS = 15
RESIDUAL_MIN_TYPE_HISTORY = 100
RESIDUAL_BANDWIDTH_KM = 0.75
RESIDUAL_HALF_LIFE_MONTHS = 12.0
RESIDUAL_SHRINKAGE = 40.0
RESIDUAL_MAX_ABS_DELTA = 0.035


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    total = weights.sum()
    if total <= 0:
        return np.nan
    index = int(np.searchsorted(np.cumsum(weights), total * 0.5, side="left"))
    return float(values[min(index, len(values) - 1)])


def _recency_weights(dates, cutoff: pd.Timestamp) -> np.ndarray:
    age_months = np.maximum(
        0.0,
        (cutoff - pd.to_datetime(dates)).dt.total_seconds().to_numpy(float)
        / (30.4375 * 24.0 * 3600.0),
    )
    return np.exp(-np.log(2.0) * age_months / RESIDUAL_HALF_LIFE_MONTHS)


def _spatial_correction(history: pd.DataFrame, query: pd.DataFrame) -> np.ndarray:
    """Return per-query log-price corrections from earlier out-of-fold errors.

    For each query home, the correction compares a distance/recency-weighted
    median residual of same-type homes within 1.5 km with the whole-type median
    residual. The local difference is shrunk by effective sample size, clipped
    to +/-0.035 log points, and applied as a multiplicative price adjustment.
    """
    query = query.reset_index(drop=True)
    out = np.zeros(len(query), dtype=float)
    if history.empty:
        return out

    history = history.copy()
    history["residual"] = np.log(history["sold_price"] / history["pred"])
    history = history[np.isfinite(history["residual"])]

    for property_type, index in query.groupby("house_type_name", sort=False).groups.items():
        positions = np.asarray(list(index), dtype=int)
        q = query.iloc[positions]
        type_history = history[history["house_type_name"] == property_type]
        if len(type_history) < RESIDUAL_MIN_TYPE_HISTORY:
            continue

        cutoffs = q["date_listed"].dt.to_period("M").dt.start_time
        max_cutoff = cutoffs.max()
        history_at_max = type_history[
            pd.to_datetime(type_history["date_sold"]) < max_cutoff
        ]
        good = (
            np.isfinite(history_at_max["latitude"])
            & np.isfinite(history_at_max["longitude"])
            & history_at_max["latitude"].between(40, 50)
            & history_at_max["longitude"].between(-90, -60)
        )
        history_at_max = history_at_max[good]
        if len(history_at_max) < RESIDUAL_MIN_NEIGHBORS:
            continue

        coordinates = _project_coordinates(
            history_at_max["latitude"], history_at_max["longitude"]
        )
        tree = cKDTree(coordinates)
        query_coordinates = _project_coordinates(q["latitude"], q["longitude"])
        sold_dates = pd.to_datetime(history_at_max["date_sold"]).to_numpy()
        residuals = history_at_max["residual"].to_numpy(float)
        center_cache: dict[pd.Timestamp, float] = {}

        for row in range(len(q)):
            cutoff = cutoffs.iloc[row]
            candidates = tree.query_ball_point(
                query_coordinates[row], r=RESIDUAL_MAX_DISTANCE_KM
            )
            if len(candidates) < RESIDUAL_MIN_NEIGHBORS:
                continue
            candidates = np.asarray(candidates, dtype=int)
            candidates = candidates[sold_dates[candidates] < cutoff]
            if len(candidates) < RESIDUAL_MIN_NEIGHBORS:
                continue

            distances = np.hypot(
                coordinates[candidates, 0] - query_coordinates[row, 0],
                coordinates[candidates, 1] - query_coordinates[row, 1],
            )
            nearest = np.argsort(distances, kind="mergesort")[:RESIDUAL_MAX_NEIGHBORS]
            candidates, distances = candidates[nearest], distances[nearest]
            weights = np.exp(-((distances / RESIDUAL_BANDWIDTH_KM) ** 2) / 2.0)
            weights *= _recency_weights(
                history_at_max["date_sold"].iloc[candidates], cutoff
            )
            if weights.sum() <= 0:
                continue

            if cutoff not in center_cache:
                earlier = history_at_max[
                    pd.to_datetime(history_at_max["date_sold"]) < cutoff
                ]
                if len(earlier) < RESIDUAL_MIN_TYPE_HISTORY:
                    center_cache[cutoff] = np.nan
                else:
                    center_cache[cutoff] = _weighted_median(
                        earlier["residual"].to_numpy(float),
                        _recency_weights(earlier["date_sold"], cutoff),
                    )
            type_center = center_cache[cutoff]
            if not np.isfinite(type_center):
                continue

            local_median = _weighted_median(residuals[candidates], weights)
            effective_n = weights.sum() ** 2 / (weights**2).sum()
            delta = (local_median - type_center) * effective_n / (
                effective_n + RESIDUAL_SHRINKAGE
            )
            out[positions[row]] = float(
                np.clip(delta, -RESIDUAL_MAX_ABS_DELTA, RESIDUAL_MAX_ABS_DELTA)
            )
    return out


def _fit_v2b_month(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    """Fit a cheap V2b model on ``train`` and predict ``valid`` (used for OOF)."""
    X_train = OffMarketFeatureModelV2b.make_features(train)
    X_valid = OffMarketFeatureModelV2b.make_features(valid)
    model = create_hgbr(max_iter=200)
    model.fit(X_train, np.log(train["sold_price"].to_numpy(dtype=float)))
    return np.exp(np.clip(model.predict(X_valid), 5, 20))


class OffMarketFeatureModelV3Residual(OffMarketFeatureModelV3):
    """Off-market V3 plus a listing-time spatial residual correction.

    The correction is a post-model layer in the spirit of the strongest public
    R151 solution. It needs earlier out-of-fold (OOF) errors, so ``_train``
    first fits the V3 model, then creates OOF residuals with six-month
    chronological blocks of a cheap V2b model. Each block model is trained on
    sales before the block and predicts that block; no block sees its own labels.

    At prediction time, residuals are read only from sales before the query
    home's **listing month**. The local correction therefore does not use sales
    that happened after the home was listed, even if the home sold months later.

    Parameters are deliberately conservative: 1.5 km search radius, 0.75 km
    Gaussian bandwidth, 12-month recency half-life, at most 50 neighbours, at
    least 15 local neighbours and 100 same-type historical rows, shrinkage 40,
    and a +/-0.035 log cap. The correction changes the price by at most about
    3.5%.

    Validation
    ----------
    Dec 2025-May 2026 walk-forward, 10,069 homes: MdAPE 7.7692%, MAPE 10.1804%,
    median signed error +0.07%. This improves the V3 model (7.8489%) by about
    0.08 pp, but the residual signal is calibrated from a V2b proxy rather than
    from V3's own OOF errors; treat it as a development result.

    Cost
    ----
    About 50 s per monthly refit on an idle machine: the V3 model (~25 s) plus
    nine block OOF V2b fits (~24 s) and the correction itself (~2 s). Concurrent
    HGBR jobs oversubscribe OpenMP threads and can multiply these times several-fold. The OOF work is repeated for
    every backtest month; a production run trains it only once.
    """

    NAME = "Off-Market V3 + residual"
    ENABLED = True
    MODEL_TAG = "HGBR"

    def __init__(self):
        super().__init__()
        self.oof_residuals: pd.DataFrame | None = None

    def _build_oof_residuals(self, data: pd.DataFrame) -> pd.DataFrame:
        start = pd.Timestamp("2022-01-01")
        end = pd.Timestamp(data["date_sold"].max()).normalize() + pd.DateOffset(months=1)
        blocks = []
        cursor = start
        while cursor < end:
            block_end = cursor + pd.DateOffset(months=6)
            train = data[data["date_sold"] < cursor]
            valid = data[(data["date_sold"] >= cursor) & (data["date_sold"] < block_end)]
            if len(train) >= 500 and len(valid) > 0:
                prediction = _fit_v2b_month(train, valid)
                blocks.append(
                    pd.DataFrame(
                        {
                            "date_sold": valid["date_sold"].to_numpy(),
                            "house_type_name": valid["house_type_name"].astype(str).to_numpy(),
                            "latitude": valid["latitude"].to_numpy(float),
                            "longitude": valid["longitude"].to_numpy(float),
                            "sold_price": valid["sold_price"].to_numpy(float),
                            "pred": prediction,
                        }
                    )
                )
            cursor = block_end
        if not blocks:
            return pd.DataFrame(
                columns=[
                    "date_sold", "house_type_name", "latitude", "longitude",
                    "sold_price", "pred",
                ]
            )
        return pd.concat(blocks, ignore_index=True)

    def _train(self, data: pd.DataFrame) -> "OffMarketFeatureModelV3Residual":
        super()._train(data)
        self.oof_residuals = self._build_oof_residuals(data)
        return self

    def _predict(self, homes: pd.DataFrame) -> pd.Series:
        base = super()._predict(homes).to_numpy(float)
        if self.oof_residuals is None or self.oof_residuals.empty:
            return pd.Series(base, index=homes.index)
        frame = homes[
            ["house_type_name", "latitude", "longitude", "date_listed"]
        ].reset_index(drop=True)
        delta = _spatial_correction(self.oof_residuals, frame)
        return pd.Series(base * np.exp(delta), index=homes.index)
