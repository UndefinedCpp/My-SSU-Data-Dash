import abc
import pandas as pd
from typing import ClassVar, Literal
import numpy as np

Track = Literal["off_market", "list_aware"]


class BaseModel(abc.ABC):
    """
    Abstract base class for all models in the SSU Data Dash project. Provides
    a unified interface for all models and ensures a consistent way to use them.

    Do not subclass this directly: use `BaseOffMarketModel` or `BaseListAwareModel`.
    """
    NAME: ClassVar[str] = ""
    TRACK: ClassVar[Track]
    ENABLED: ClassVar[bool] = False
    MODEL_TAG: ClassVar[str] = "model"

    def train(self, data: pd.DataFrame) -> 'BaseModel':
        """
        Train the model with the provided data.
        """
        return self._train(self._prepare(data)) or self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Make predictions using the trained model on the provided data.
        """
        x = self._predict(self._prepare(data))
        if isinstance(x, pd.Series):
            return x
        elif isinstance(x, np.ndarray):
            return pd.Series(x)
        else:
            raise ValueError(f'Unrecognized prediction result type: {type(x)}')

    def _prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        """Restrict data to the columns available on this model's track."""
        return data

    @abc.abstractmethod
    def _train(self, data: pd.DataFrame) -> 'BaseModel | None':
        """Subclass hook: fit the model. May return self or None."""

    @abc.abstractmethod
    def _predict(self, data: pd.DataFrame) -> pd.Series:
        """Subclass hook: return one predicted sold price per row."""

    def get_id(self) -> str:
        return self.__class__.__name__


class BaseOffMarketModel(BaseModel):
    """Values homes that have no list price; `list_price` is hidden from the model."""
    TRACK: ClassVar[Track] = "off_market"

    def _prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.drop(columns=["list_price"], errors="ignore")


class BaseListAwareModel(BaseModel):
    """Values homes that have a list price, which is available as `list_price`."""
    TRACK: ClassVar[Track] = "list_aware"
