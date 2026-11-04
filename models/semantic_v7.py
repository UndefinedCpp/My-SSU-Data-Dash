"""Frozen language representations plus strictly historical quality signals."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from models.features import square_feet
from models.hybrid_v5 import ListAwareValueGapHGBV5
from models.robust_v6 import OffMarketRobustXGBV6

MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
REVISION = 'c9745ed1d9f207416be6d2e6f8de32d1f16199bf'


class FrozenSentenceEncoder:
    _vectors = None
    _index = None
    _encoder = None

    def transform(self, homes):
        texts = homes.remarks.fillna('').astype(str).tolist()
        directory = Path(__file__).resolve().parents[1] / 'data/semantic'
        cls = type(self)
        if cls._vectors is None and (directory / 'metadata.json').exists():
            metadata = json.loads((directory / 'metadata.json').read_text())
            if (metadata['complete_rows'] == metadata['rows'] and metadata['revision'] == REVISION
                    and metadata.get('normalized') and metadata.get('max_seq_length') == 256):
                cls._vectors = np.load(directory / 'vectors.npy', mmap_mode='r')
                hashes = np.load(directory / 'hashes.npy')
                cls._index = {value.decode(): i for i, value in enumerate(hashes)}
        hashes = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        result = np.empty((len(texts), 384), dtype=np.float32)
        missing = []
        for i, digest in enumerate(hashes):
            if cls._index is not None and digest in cls._index:
                result[i] = cls._vectors[cls._index[digest]]
            else:
                missing.append(i)
        if missing:
            if cls._encoder is None:
                import torch
                from sentence_transformers import SentenceTransformer
                local = directory / 'encoder'
                source = str(local) if local.exists() else MODEL_ID
                cls._encoder = SentenceTransformer(source, revision=REVISION,
                    device='mps' if torch.backends.mps.is_available() else 'cpu')
                cls._encoder.max_seq_length = 256
            result[missing] = cls._encoder.encode([texts[i] for i in missing], batch_size=128,
                                                  normalize_embeddings=True, show_progress_bar=False)
        return result


class HistoricalSemanticSignal:
    def __init__(self, list_aware=False, min_history=1000):
        self.list_aware = list_aware
        self.min_history = min_history
        self.snapshots = {}

    def _fit(self, history, vectors):
        if len(history) < self.min_history:
            return None
        target = np.log(history.sold_price.to_numpy(float))
        if self.list_aware:
            target -= np.log(history.list_price.to_numpy(float))
        else:
            # Strip broad property/neighbourhood and market levels before asking
            # language to estimate the remaining quality premium.
            target -= np.log(square_feet(history.house_area).to_numpy(float))
            group = history.house_type_name.astype(str) + '|' + history.community.astype(str)
            target = pd.Series(target, index=history.index)
            target = target - target.groupby(group).transform('median')
            target = target - target.groupby(history.date_listed.dt.to_period('Q')).transform('median')
        model = Ridge(alpha=10, solver='cholesky')
        model.fit(vectors, np.asarray(target))
        return model

    def fit_transform(self, sales, vectors):
        cutoffs = sales.date_listed.map(lambda date: pd.Timestamp(
            year=date.year, month=1 if date.month <= 6 else 7, day=1))
        values = np.zeros(len(sales), dtype=float)
        self.snapshots = {}
        for cutoff in sorted(cutoffs.unique()):
            mask = sales.date_sold.lt(cutoff).to_numpy()
            model = self._fit(sales.loc[mask], vectors[mask])
            self.snapshots[cutoff] = model
            positions = np.flatnonzero(cutoffs.eq(cutoff).to_numpy())
            if model is not None:
                values[positions] = model.predict(vectors[positions])
        cutoff = sales.date_sold.max().normalize() + pd.Timedelta(days=1)
        self.snapshots[cutoff] = self._fit(sales, vectors)
        return values

    def transform(self, homes, vectors):
        if not self.snapshots:
            raise RuntimeError('Fit semantic history before transform')
        values = np.zeros(len(homes), dtype=float)
        for month, positions in homes.groupby(homes.date_listed.dt.to_period('M')).indices.items():
            eligible = [cutoff for cutoff in self.snapshots if cutoff <= month.start_time]
            if eligible and self.snapshots[max(eligible)] is not None:
                values[positions] = self.snapshots[max(eligible)].predict(vectors[positions])
        return values


class _SemanticFeatures:
    def __init__(self, iterations=None):
        super().__init__(**({'iterations': iterations} if iterations is not None else {}))
        self.encoder = FrozenSentenceEncoder()
        self.semantic_signal = HistoricalSemanticSignal(list_aware=self.LIST_AWARE)
        self.pca = None

    def _features(self, homes, fit=False):
        frame = super()._features(homes, fit=fit)
        vectors = self.encoder.transform(homes)
        if fit:
            self.pca = PCA(n_components=min(32, len(homes)-1), random_state=42, svd_solver='randomized')
            components = self.pca.fit_transform(vectors)
            quality = self.semantic_signal.fit_transform(homes, vectors)
        else:
            components = self.pca.transform(vectors)
            quality = self.semantic_signal.transform(homes, vectors)
        semantic = pd.DataFrame(components, index=homes.index,
                                 columns=[f'semantic_{i}' for i in range(components.shape[1])])
        semantic['historical_semantic_quality'] = quality
        return pd.concat([frame, semantic], axis=1)


class OffMarketSemanticV7(_SemanticFeatures, OffMarketRobustXGBV6):
    NAME = 'Off-Market Semantic V7'
