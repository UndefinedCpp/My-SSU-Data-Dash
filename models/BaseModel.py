import abc
from typing import ClassVar, Literal
import numpy as np
import pandas as pd

Track = Literal["off_market", "list_aware"]


class BaseModel(abc.ABC):
    NAME: ClassVar[str] = ""
    TRACK: ClassVar[Track]
    ENABLED: ClassVar[bool] = False
    MODEL_TAG: ClassVar[str] = "model"

    def train(self, data: pd.DataFrame) -> "BaseModel":
        return self._train(self._prepare(data)) or self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        prediction = self._predict(self._prepare(data))
        if isinstance(prediction, pd.Series):
            return prediction
        if isinstance(prediction, np.ndarray):
            return pd.Series(prediction)
        raise ValueError(f"Unrecognized prediction result type: {type(prediction)}")

    def _prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    @abc.abstractmethod
    def _train(self, data: pd.DataFrame) -> "BaseModel | None":
        ...

    @abc.abstractmethod
    def _predict(self, homes: pd.DataFrame) -> pd.Series:
        ...

    def get_id(self) -> str:
        return self.__class__.__name__


class BaseOffMarketModel(BaseModel):
    TRACK: ClassVar[Track] = "off_market"

    def _prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.drop(columns=["list_price"], errors="ignore")


class BaseListAwareModel(BaseModel):
    TRACK: ClassVar[Track] = "list_aware"
