"""Equal log-price blends to reduce disagreement between tree learners."""
import numpy as np
import pandas as pd

from models.BaseModel import BaseListAwareModel, BaseOffMarketModel
from models.attempt1 import ListAwareBuildingV1, OffMarketBuildingXGBoostV1
from models.catboost_v2 import OffMarketCatBoostV2


class _GeometricBlend:
    MODEL_TAG = 'Ensemble'
    ENABLED = True

    def __init__(self):
        self.baseline = self.BASELINE()
        self.candidate = self.CANDIDATE()

    def _train(self, data):
        self.baseline.train(data)
        self.candidate.train(data)
        return self

    def _predict(self, homes):
        first = np.asarray(self.baseline.predict(homes))
        second = np.asarray(self.candidate.predict(homes))
        return pd.Series(np.exp(0.5 * (np.log(first) + np.log(second))), index=homes.index)


class OffMarketEnsembleV2(_GeometricBlend, BaseOffMarketModel):
    NAME = 'Off-Market Ensemble V2'
    BASELINE = OffMarketBuildingXGBoostV1
    CANDIDATE = OffMarketCatBoostV2
