from typing import Callable

from sklearn.ensemble import HistGradientBoostingRegressor
from models.BaseModel import BaseListAwareModel, BaseOffMarketModel
import numpy as np
import pandas as pd

PROPERTY_TYPES = [
    "Condo Apt",
    "Detached",
    "Semi-Detached",
    "Condo Townhouse",
    "Freehold Townhouse",
    "Co-Op Apt",
]

def square_feet(house_area):
    """house_area is text, either one number ("622") or a range ("900-1099").
    Return a number: the number itself, or the middle of the range."""
    parts = house_area.astype(str).str.split("-", expand=True).reindex(columns=[0, 1])
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    return (low + high) / 2


class SimpleOffMarketFeatureModel(BaseOffMarketModel):
    """
    Simple off-market model using basic features and HistGradientBoostingRegressor.
    """

    NAME = "Simple Off-Market Feature Model"
    ENABLED = True
    MODEL_TAG = "model"

    def __init__(self):
        super().__init__()
        self.model = HistGradientBoostingRegressor()

    @staticmethod
    def make_features(homes: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=homes.index)
        X["area"] = square_feet(homes["house_area"])
        X["bedrooms"] = homes["bedrooms"]
        X["washrooms"] = homes["washrooms"]
        X["parking"] = homes["parking_total"]
        X["latitude"] = homes["latitude"]
        X["longitude"] = homes["longitude"]
        # Months since January 2021, so the model can follow prices over time.
        X["months"] = (
            homes["date_listed"] - pd.Timestamp("2021-01-01")
        ).dt.days / 30.44
        # One 0/1 column per property type.
        for kind in PROPERTY_TYPES:
            X["is_" + kind] = (homes["house_type_name"] == kind).astype(int)
        return X

    def _train(self, homes: pd.DataFrame) -> "SimpleOffMarketFeatureModel":
        X = self.make_features(homes)
        y = np.log(homes["sold_price"])
        self.model.fit(X, y)
        return self

    def _predict(self, homes: pd.DataFrame) -> pd.Series:
        X = self.make_features(homes)
        log_preds = self.model.predict(X)
        return np.exp(log_preds)


class SimpleListAwareFeatureModel(BaseListAwareModel):

    NAME = "Simple List-Aware Feature Model"
    ENABLED = True
    MODEL_TAG = "model"

    def __init__(self):
        super().__init__()
        self.model = HistGradientBoostingRegressor()

    @staticmethod
    def make_features(homes: pd.DataFrame) -> pd.DataFrame:
        X = SimpleOffMarketFeatureModel.make_features(homes)
        X["list_price"] = homes["list_price"]
        return X

    def _train(self, homes: pd.DataFrame) -> "SimpleListAwareFeatureModel":
        X = self.make_features(homes)
        y = np.log(homes["sold_price"])
        self.model.fit(X, y)
        return self

    def _predict(self, homes: pd.DataFrame) -> pd.Series:
        X = self.make_features(homes)
        log_preds = self.model.predict(X)
        return np.exp(log_preds)
