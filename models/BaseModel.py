import abc
import pandas as pd
from typing import ClassVar

class BaseModel(abc.ABC):
    """
    Abstract base class for all models in the SSU Data Dash project. Provides
    a unified interface for all models and ensures a consistent way to use them.
    """
    NAME: ClassVar[str] = ""
    TRACKS: ClassVar[tuple[str, ...]] = ("off_market", "list_aware")
    ENABLED: ClassVar[bool] = True
    MODEL_TAG: ClassVar[str] = "model"


    @abc.abstractmethod
    def train(self, data: pd.DataFrame) -> 'BaseModel':
        """
        Train the model with the provided data.
        """
        pass

    @abc.abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Make predictions using the trained model on the provided data.
        """
        pass

    def get_id(self) -> str:
        return self.__class__.__name__