import unicodedata
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import SplineTransformer


PROPERTY_TYPES = [
    "Condo Apt",
    "Detached",
    "Semi-Detached",
    "Condo Townhouse",
    "Freehold Townhouse",
    "Co-Op Apt",
]

HOUSE_STYLES = [
    "Apartment", "Bungalow", "2-Storey", "3-Storey", "2 1/2 Storey",
    "Bungalow-Raised", "Loft", "Stacked Townhse", "Backsplit 4",
    "Sidesplit 3", "Multi-Level", "Backsplit 3", "1 1/2 Storey",
    "Sidesplit 4", "1 Storey/Apt", "Backsplit 5",
]

MUNICIPALITIES = ["Toronto", "North York", "Scarborough", "Etobicoke", "York"]

TEXT_CUE_WORDBANK = {
    "text:renovated": ["renovated", "renovation", "updated", "upgraded", "refreshed", "redone"],
    "text:original": ["original", "dated", "as-is", "worn", "tired", "outdated"],
    "text:potential": ["potential", "investor"],
    "text:luxury": ["luxury", "luxurious", "stunning", "high-end", "premium"],
    "text:view": ["view", "skyline", "ravine", "waterfront"],
}


def square_feet(house_area: pd.Series) -> pd.Series:
    """Parse a square-foot number or range to its midpoint."""
    parts = house_area.astype(str).str.split("-", expand=True).reindex(columns=[0, 1])
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    return (low + high) / 2


def _normalize_remarks(value: str) -> str:
    return unicodedata.normalize("NFKD", value.strip().lower())


def base_features(homes: pd.DataFrame, *, list_aware: bool = False) -> pd.DataFrame:
    """Property/geography features shared by both final models."""
    X = pd.DataFrame(index=homes.index)
    X["area"] = square_feet(homes["house_area"])
    X["bedrooms"] = homes["bedrooms"]
    X["washrooms"] = homes["washrooms"]
    X["parking"] = homes["parking_total"]
    X["latitude"] = homes["latitude"]
    X["longitude"] = homes["longitude"]
    X["months"] = (homes["date_listed"] - pd.Timestamp("2021-01-01")).dt.days / 30.44
    for kind in PROPERTY_TYPES:
        X["is_" + kind] = (homes["house_type_name"] == kind).astype(int)

    X["bed_per_bath"] = homes["bedrooms"] / homes["washrooms"].replace(0, 1)
    X["lot_area"] = homes["lot_front"] * homes["lot_depth"]
    dx = (homes["longitude"] + 79.38) * 111 * 0.72
    dy = (homes["latitude"] - 43.72) * 111
    X["distance_center"] = np.sqrt(dx**2 + dy**2)

    for style in HOUSE_STYLES:
        X["house_style:" + style] = (homes["house_style"] == style).astype(int)
    for muni in MUNICIPALITIES:
        X["municipality:" + muni] = (homes["municipality"] == muni).astype(int)

    X["community"] = homes["community"].astype(str)
    X["month_of_year"] = homes["date_listed"].dt.month

    remarks = homes["remarks"].fillna("").astype(str).apply(_normalize_remarks)
    for cue, keywords in TEXT_CUE_WORDBANK.items():
        X[cue] = remarks.apply(lambda text: sum(text.count(k) for k in keywords))

    if list_aware:
        X["list_price"] = homes["list_price"]
        X["log_list_price"] = np.log(homes["list_price"])
    return X


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


def _building_area(series: pd.Series) -> np.ndarray:
    return square_feet(series).to_numpy(dtype=float)


def _prepare_building_history(frame: pd.DataFrame, include_ratio: bool) -> pd.DataFrame:
    columns = [
        "date_listed", "date_sold", "sold_price", "house_type_name",
        "bedrooms", "washrooms", "parking_total", "house_area", "latitude",
        "longitude", "days_on_market",
    ]
    if include_ratio:
        columns.append("list_price")
    prepared = frame[columns].copy()
    prepared["_area"] = _building_area(prepared["house_area"])
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
        self.history = _prepare_building_history(train, self.include_ratio)
        return self._transform_by_month(train, self.history)

    def transform(self, homes: pd.DataFrame) -> pd.DataFrame:
        if self.history is None:
            raise RuntimeError("Call fit_transform before transform")
        return self._transform_by_month(homes, self.history)

