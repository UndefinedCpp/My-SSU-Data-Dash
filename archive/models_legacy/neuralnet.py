from models.BaseModel import BaseListAwareModel, BaseOffMarketModel
from models.featuremodel_v2b import OffMarketFeatureModelV2b
import numpy as np
from typing import Callable, Any
from pathlib import Path
from loguru import logger
from tqdm import trange
import pandas as pd

# Import PyTorch if available
try:
    import torch
    import torch.nn
    import torch.optim
except ImportError:
    HAS_PYTORCH_INSTALLED = False
    logger.warning(
        "PyTorch is not installed, neural network models will not be available."
    )
    logger.warning("Install PyTorch with `pip install torch==2.12.1`.")
    logger.warning(
        "Installation instructions can be found at https://pytorch.org/get-started/locally/, and the process might be a bit more complicated on Windows."
    )
else:
    HAS_PYTORCH_INSTALLED = True
    logger.success(f"PyTorch {torch.__version__} is available")


COMMUNITY_NAMES_PATH = Path(__file__).resolve().parents[1] / "community_names.txt"
with COMMUNITY_NAMES_PATH.open() as f:
    COMMUNITY_NAME_VOCABULARY = [
        line.strip() for line in f if line.strip()
    ]

if len(COMMUNITY_NAME_VOCABULARY) != len(set(COMMUNITY_NAME_VOCABULARY)):
    raise ValueError(f"Duplicate community names in {COMMUNITY_NAMES_PATH}")


def build_ffn_features(
    homes: pd.DataFrame, extra_features: pd.DataFrame | None = None
) -> np.ndarray:
    """Turn a DataFrame into the dense float32 matrix the FFN expects.

    Starts from the V2b feature set, standardises the continuous columns, adds
    `extra_features` (used for the list price on the list-aware track) and
    one-hot encodes community.
    """
    x = OffMarketFeatureModelV2b.make_features(homes)
    if x.shape[1] != 44:
        raise ValueError(f"Expected 44 V2b features, got {x.shape[1]}")
    # Make a flag to indicate missing lot area values. The feature is missing
    # structurely at around 58% of the data.
    X = x.copy()
    X["lot_area_missing"] = X["lot_area"].isna().astype(np.float32)
    # Normalize some continuous features using standardization
    X["area"] = (np.log(X["area"]) - 7.0825830638) / 0.5156774522
    X["lot_area"] = (
        (np.log(X["lot_area"]) - 8.2308273733) / 0.6059667653
    ).fillna(0.0)
    X["latitude"] = (X["latitude"] - 43.6434524242) / 0.0543678104
    X["longitude"] = (X["longitude"] - (-79.3786543046)) / 0.1005938192
    X["months"] = (X["months"] - 27.4502227332) / 19.1933506751
    X["distance_center"] = (
        X["distance_center"] - 12.1517646311
    ) / 5.0589512268
    if extra_features is not None:
        X = pd.concat([X, extra_features], axis=1)
    # One-hot encode community feature
    community = pd.Series(
        pd.Categorical(
            X.pop("community"), categories=COMMUNITY_NAME_VOCABULARY
        ),
        index=X.index,
    )
    community_features = pd.get_dummies(community, dtype=np.float32)
    X = pd.concat([X, community_features], axis=1)
    X = X.to_numpy(dtype=np.float32)
    assert np.all(np.isfinite(X)), "Non-finite values found in feature matrix"
    return X


class SimpleFeedForwardModel:

    def __init__(
        self,
        layer_size: list[int],
        act=torch.nn.ReLU,
        loss_fn=torch.nn.MSELoss,
        feature_maker: Callable[[Any], np.ndarray | torch.Tensor] = None,
    ):
        """
        Create a feed-forward neural network with the specified layer sizes.
        The output layer will have size 1 and no activation function applied to it.
        """
        self.network = torch.nn.Sequential()
        for i in range(len(layer_size) - 1):
            self.network.append(torch.nn.Linear(layer_size[i], layer_size[i + 1]))
            self.network.append(act())
        self.network.append(torch.nn.Linear(layer_size[-1], 1))
        self.feature_maker = feature_maker
        self.loss_fn = loss_fn
        # Filled in by train(); the target is standardised during training and
        # converted back to a price in SimpleOffMarketFFN._predict.
        self.y_mean = None
        self.y_std = None

    def forward(self, x):
        if self.feature_maker is not None:
            x = self.feature_maker(x)
        x = torch.as_tensor(x, dtype=torch.float32)
        return self.network(x)

    def predict_raw(self, x):
        """Return predictions in the units of the target train() was given."""
        with torch.no_grad():
            output = self.forward(x).numpy().reshape(-1)
        return output * self.y_std + self.y_mean

    def train(
        self, x, y, epochs=30, lr=1e-3, batch_size=512, verbose=True
    ):
        if self.feature_maker is not None:
            x = self.feature_maker(x)
        # np.asarray accepts the pandas Series that _train passes in; torch.tensor
        # raises "could not determine the shape of object type 'Series'".
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        # The target is log price, around 13.7. Standardising it lets the network
        # start near the mean instead of near zero, which otherwise takes far more
        # optimiser steps than we run.
        self.y_mean = float(y.mean())
        self.y_std = float(y.std()) or 1.0
        y = torch.as_tensor((y - self.y_mean) / self.y_std, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        loss_fn = self.loss_fn()
        n = len(x)

        if verbose:
            logger.info(
                f"Training started, {lr=}, {epochs=}, {batch_size=}, x.shape="
                f"{tuple(x.shape)}, y.shape={tuple(y.shape)}"
            )

        for epoch in range(epochs):
            permutation = torch.randperm(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                batch = permutation[start : start + batch_size]
                optimizer.zero_grad()
                loss: torch.Tensor = loss_fn(self.network(x[batch]), y[batch])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch)
            if verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs}, loss: {epoch_loss / n:.4f}  ",
                    end="\r",
                )

        if verbose:
            print()
            logger.info("Training completed.")


class SimpleOffMarketFFN(BaseOffMarketModel):

    NAME = "Simple Off-Market FFN"
    ENABLED = True
    MODEL_TAG = "neural"

    # Six small networks averaged together. A single [64, 64] net still underfits
    # at 30 epochs, while wider nets overfit ([256, 128] reached ~7.8 MdAPE on the
    # training split but ~10.2 on validation), so averaging a mix of widths is the
    # configuration that gets closest to the boosted trees. Seeds keep benchmark
    # runs repeatable.
    HIDDEN_LAYERS = [(64, 64), (64, 64), (64, 64), (128, 64), (128, 64), (128, 64)]

    def __init__(self):
        super().__init__()
        self.ffns = []

    @staticmethod
    def feature_maker(x: pd.DataFrame) -> np.ndarray:
        X = build_ffn_features(x)
        # Ensure the shape and data is correct
        if X.shape[1] != 189:
            raise ValueError(f"Expected 189 model features, got {X.shape[1]}")
        return X

    def _train(self, data):
        # Build the feature matrix once and share it across the ensemble.
        X = self.feature_maker(data)
        y = np.log(data["sold_price"].to_numpy())
        self.ffns = []
        for seed, hidden in enumerate(self.HIDDEN_LAYERS):
            torch.manual_seed(seed)
            ffn = SimpleFeedForwardModel(layer_size=[189, *hidden])
            ffn.train(X, y, verbose=False)
            self.ffns.append(ffn)
        return self

    def _predict(self, data):
        X = self.feature_maker(data)
        # Average in log-price space, then convert back to a price.
        log_price = np.mean([ffn.predict_raw(X) for ffn in self.ffns], axis=0)
        return np.exp(log_price)


class SimpleListAwareFFN(BaseListAwareModel):

    NAME = "Simple List-Aware FFN"
    ENABLED = True
    MODEL_TAG = "neural"

    # Mean and standard deviation of log(list_price)
    LOG_LIST_PRICE_MEAN = 13.7286333076
    LOG_LIST_PRICE_STD = 0.5202904113

    # Same ensemble as the off-market track, see the comment there.
    HIDDEN_LAYERS = [(64, 64), (64, 64), (64, 64), (128, 64), (128, 64), (128, 64)]

    def __init__(self):
        super().__init__()
        self.ffns = []

    @staticmethod
    def make_list_price_features(homes: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "log_list_price": (
                    np.log(homes["list_price"].to_numpy())
                    - SimpleListAwareFFN.LOG_LIST_PRICE_MEAN
                )
                / SimpleListAwareFFN.LOG_LIST_PRICE_STD
            },
            index=homes.index,
        )

    @staticmethod
    def feature_maker(x: pd.DataFrame) -> np.ndarray:
        X = build_ffn_features(
            x, SimpleListAwareFFN.make_list_price_features(x)
        )
        if X.shape[1] != 190:
            raise ValueError(f"Expected 190 model features, got {X.shape[1]}")
        return X

    def _train(self, data):
        # Target: the relative correction against the list price,
        # (sold - list) / list. It is centred on zero with a std of ~0.11, so the
        # network only has to learn the small adjustment around the list price.
        X = self.feature_maker(data)
        y = (data["sold_price"] / data["list_price"] - 1).to_numpy()
        self.ffns = []
        for seed, hidden in enumerate(self.HIDDEN_LAYERS):
            torch.manual_seed(seed)
            ffn = SimpleFeedForwardModel(layer_size=[190, *hidden])
            ffn.train(X, y, verbose=False)
            self.ffns.append(ffn)
        return self

    def _predict(self, data):
        X = self.feature_maker(data)
        correction = np.mean([ffn.predict_raw(X) for ffn in self.ffns], axis=0)
        return data["list_price"].to_numpy(dtype=float) * (1.0 + correction)
