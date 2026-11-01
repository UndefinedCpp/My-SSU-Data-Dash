import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

from models.building_v1 import ListAwareBuildingV1, OffMarketBuildingV1
from models.featuremodel_v2b import ListAwareFeatureModelV2b

HAS_PYTORCH_INSTALLED = torch is not None


class DenseFeaturePreprocessor:
    """Median-impute, standardise and one-hot encode a tabular feature frame."""

    def __init__(self, max_z: float = 8.0):
        self.max_z = max_z
        self.columns_: list[str] = []
        self.communities_: list[str] = []
        self.medians_: pd.Series | None = None
        self.means_: pd.Series | None = None
        self.stds_: pd.Series | None = None
        self.missing_columns_: list[str] = []
        self.input_dim_: int | None = None

    def _numeric(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame[self.columns_].astype(float)

    def _assemble(
        self, numeric: pd.DataFrame, community: pd.Series
    ) -> np.ndarray:
        assert self.medians_ is not None and self.means_ is not None
        assert self.stds_ is not None
        missing = numeric[self.missing_columns_].isna().to_numpy(dtype=np.float32)
        filled = numeric.fillna(self.medians_)
        z = (filled - self.means_) / self.stds_
        z = z.clip(-self.max_z, self.max_z).fillna(0.0)
        onehot = pd.get_dummies(
            pd.Categorical(community, categories=self.communities_),
            dtype=np.float32,
        )
        onehot = onehot.reindex(columns=self.communities_, fill_value=0)
        matrix = np.hstack(
            [
                z.to_numpy(dtype=np.float32),
                missing,
                onehot.to_numpy(dtype=np.float32),
            ]
        )
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        self.input_dim_ = matrix.shape[1]
        return matrix

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        frame = frame.copy()
        community = frame.pop("community").astype(str)
        self.columns_ = list(frame.columns)
        self.communities_ = sorted(community.unique().tolist())
        numeric = self._numeric(frame)
        self.medians_ = numeric.median().fillna(0.0)
        self.means_ = numeric.mean().fillna(0.0)
        self.stds_ = numeric.std().replace(0.0, 1.0).fillna(1.0)
        self.missing_columns_ = [
            column for column in self.columns_ if numeric[column].isna().any()
        ]
        return self._assemble(numeric, community)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.columns_:
            raise RuntimeError("Call fit_transform before transform")
        community = frame["community"].astype(str)
        return self._assemble(self._numeric(frame), community)


class _TorchMLP:
    """Small MLP with a standardised regression target."""

    def __init__(self, input_dim: int, hidden: tuple[int, ...]):
        layers: list['torch.nn.Module'] = []
        previous = input_dim
        for width in hidden:
            layers.append(torch.nn.Linear(previous, width))
            layers.append(torch.nn.ReLU())
            previous = width
        layers.append(torch.nn.Linear(previous, 1))
        self.network = torch.nn.Sequential(*layers)
        self.y_mean = 0.0
        self.y_std = 1.0

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int,
        lr: float,
        batch_size: int,
    ) -> None:
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        self.y_mean = float(y.mean())
        self.y_std = float(y.std()) or 1.0
        y = torch.as_tensor(
            (y - self.y_mean) / self.y_std, dtype=torch.float32
        )
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        n = len(x)
        for _ in range(epochs):
            permutation = torch.randperm(n)
            for start in range(0, n, batch_size):
                batch = permutation[start : start + batch_size]
                optimizer.zero_grad()
                loss = loss_fn(self.network(x[batch]), y[batch])
                loss.backward()
                optimizer.step()

    def predict_raw(self, x: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        with torch.no_grad():
            output = self.network(x).numpy().reshape(-1)
        return output * self.y_std + self.y_mean


class _NeuralNetMixin:
    """Shared ensemble training for the two tracks."""

    HIDDEN_LAYERS = ((128, 128), )
    EPOCHS = 30
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-3

    def _fit_networks(self, X: np.ndarray, y: np.ndarray) -> None:
        # One thread avoids the macOS OpenMP clash between PyTorch and the
        # XGBoost/libomp stack that model discovery imports. Multi-threaded
        # training segfaults after the tabular feature builders have run.
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        self.ffns = []
        for seed, hidden in enumerate(self.HIDDEN_LAYERS):
            torch.manual_seed(seed)
            network = _TorchMLP(X.shape[1], hidden)
            network.train(
                X,
                y,
                epochs=self.EPOCHS,
                lr=self.LEARNING_RATE,
                batch_size=self.BATCH_SIZE,
            )
            self.ffns.append(network)

    def _predict_networks(self, X: np.ndarray) -> np.ndarray:
        return np.mean([ffn.predict_raw(X) for ffn in self.ffns], axis=0)


class OffMarketNeuralNetV2(_NeuralNetMixin, OffMarketBuildingV1):
    """Off-market MLP on V3 + exact-building features."""

    NAME = "Off-Market Neural Net V2 (V3 + building)"
    ENABLED = HAS_PYTORCH_INSTALLED
    MODEL_TAG = "NN"

    def __init__(self):
        super().__init__()
        self.preprocessor = DenseFeaturePreprocessor()
        self.ffns = []

    def _train(self, data: pd.DataFrame) -> "OffMarketNeuralNetV2":
        X = self.preprocessor.fit_transform(self._fit_features(data))
        y = np.log(data["sold_price"].to_numpy(dtype=float))
        self._fit_networks(X, y)
        return self

    def _predict(self, homes: pd.DataFrame) -> pd.Series:
        X = self.preprocessor.transform(self._transform_features(homes))
        raw = self._predict_networks(X)
        return pd.Series(np.exp(np.clip(raw, 5, 20)), index=homes.index)


class ListAwareNeuralNetV2(_NeuralNetMixin, ListAwareBuildingV1):
    """List-aware MLP on V2b + exact-building features and ratio target."""

    NAME = "List-Aware Neural Net V2 (building + ratio)"
    ENABLED = HAS_PYTORCH_INSTALLED
    MODEL_TAG = "NN"

    def __init__(self):
        super().__init__()
        self.preprocessor = DenseFeaturePreprocessor()
        self.ffns = []

    def _features(self, data: pd.DataFrame, *, fit: bool) -> np.ndarray:
        raw = pd.concat(
            [
                ListAwareFeatureModelV2b.make_features(data),
                self.building.fit_transform(data)
                if fit
                else self.building.transform(data),
            ],
            axis=1,
        )
        return (
            self.preprocessor.fit_transform(raw)
            if fit
            else self.preprocessor.transform(raw)
        )

    def _train(self, data: pd.DataFrame) -> "ListAwareNeuralNetV2":
        X = self._features(data, fit=True)
        y = np.log(
            data["sold_price"].to_numpy(dtype=float)
            / data["list_price"].to_numpy(dtype=float)
        )
        self._fit_networks(X, y)
        return self

    def _predict(self, homes: pd.DataFrame) -> pd.Series:
        X = self._features(homes, fit=False)
        ratio = np.exp(self._predict_networks(X))
        return pd.Series(
            homes["list_price"].to_numpy(dtype=float) * ratio,
            index=homes.index,
        )
