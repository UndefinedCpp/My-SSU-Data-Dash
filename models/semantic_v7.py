"""Frozen language representations plus strictly historical quality signals."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from models.features import square_feet
from models.robust_v6 import OffMarketRobustXGBV6

MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
REVISION = 'c9745ed1d9f207416be6d2e6f8de32d1f16199bf'


class FrozenSentenceEncoder:
    """Frozen MiniLM embeddings cached by exact text hash.

    The cache is process-wide and persisted under data/semantic. Training and
    prediction therefore encode each distinct remark once, even when several
    model instances are fitted in one backtest. The encoder keeps a whole batch
    on its accelerator until one CPU transfer; converting every mini-batch to
    NumPy is much slower on MPS.
    """

    _vectors = None
    _index = None
    _encoder = None
    _loaded_directory = None

    def __init__(self, cache_dir=None, batch_size=256):
        self.cache_dir = (Path(cache_dir) if cache_dir is not None
                          else Path(__file__).resolve().parents[1] / 'data/semantic')
        self.batch_size = batch_size

    @classmethod
    def _reset(cls):
        cls._vectors = None
        cls._index = None
        cls._encoder = None
        cls._loaded_directory = None

    def _load_cache(self):
        cls = type(self)
        directory = self.cache_dir
        if cls._loaded_directory == str(directory):
            return
        cls._vectors = None
        cls._index = None
        cls._loaded_directory = str(directory)
        metadata_path = directory / 'metadata.json'
        if not metadata_path.exists():
            return
        try:
            metadata = json.loads(metadata_path.read_text())
            if metadata.get('model_id') not in (None, MODEL_ID):
                return
            if (metadata.get('revision') != REVISION
                    or not metadata.get('normalized')
                    or metadata.get('max_seq_length') != 256):
                return
            rows = int(metadata['rows'])
            if int(metadata['complete_rows']) != rows:
                return
            vectors = np.load(directory / 'vectors.npy')
            hashes = np.load(directory / 'hashes.npy')
            if (vectors.ndim != 2 or vectors.shape[0] != rows or vectors.shape[1] != 384
                    or hashes.ndim != 1 or hashes.shape[0] != rows):
                raise ValueError('embedding cache shape does not match metadata')
            vectors = np.asarray(vectors, dtype=np.float32)
            index = {}
            for position, value in enumerate(hashes):
                digest = value.decode('ascii') if isinstance(value, (bytes, np.bytes_)) else str(value)
                if digest in index:
                    raise ValueError('duplicate text hash in embedding cache')
                index[digest] = position
            cls._vectors = np.ascontiguousarray(vectors)
            cls._index = index
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            # A stale or damaged cache must never be used for a prediction.
            cls._vectors = None
            cls._index = None

    def _encode(self, texts):
        cls = type(self)
        if cls._encoder is None:
            import torch
            from sentence_transformers import SentenceTransformer
            local = self.cache_dir / 'encoder'
            source = str(local) if local.exists() else MODEL_ID
            cls._encoder = SentenceTransformer(
                source, revision=REVISION,
                device='mps' if torch.backends.mps.is_available() else 'cpu',
            )
            cls._encoder.max_seq_length = 256
        vectors = cls._encoder.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True,
            show_progress_bar=False, convert_to_tensor=True,
        )
        if hasattr(vectors, 'detach'):
            vectors = vectors.detach().cpu().numpy()
        return np.asarray(vectors, dtype=np.float32)

    def _save_cache(self):
        cls = type(self)
        if cls._vectors is None or cls._index is None or not cls._index:
            return
        directory = self.cache_dir
        try:
            directory.mkdir(parents=True, exist_ok=True)
            hashes = np.asarray(list(cls._index), dtype='S64')
            rows = len(cls._index)
            metadata = {
                'model_id': MODEL_ID,
                'revision': REVISION,
                'normalized': True,
                'max_seq_length': 256,
                'dimension': 384,
                'rows': rows,
                'complete_rows': rows,
            }
            vectors_tmp = directory / '.vectors.npy.tmp.npy'
            hashes_tmp = directory / '.hashes.npy.tmp.npy'
            metadata_tmp = directory / '.metadata.json.tmp'
            np.save(vectors_tmp, cls._vectors, allow_pickle=False)
            np.save(hashes_tmp, hashes, allow_pickle=False)
            metadata_tmp.write_text(json.dumps(metadata) + '\n')
            # Metadata is replaced last: readers only accept a complete pair.
            vectors_tmp.replace(directory / 'vectors.npy')
            hashes_tmp.replace(directory / 'hashes.npy')
            metadata_tmp.replace(directory / 'metadata.json')
        except OSError:
            # A read-only checkout should still be usable; the cache is optional.
            pass

    def transform(self, homes):
        texts = homes.remarks.fillna('').astype(str).tolist()
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        cls = type(self)
        self._load_cache()
        hashes = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        result = np.empty((len(texts), 384), dtype=np.float32)
        missing = {}
        for position, digest in enumerate(hashes):
            cached = cls._index.get(digest) if cls._index is not None else None
            if cached is None:
                missing.setdefault(digest, texts[position])
            else:
                result[position] = cls._vectors[cached]
        if missing:
            items = list(missing.items())
            encoded = self._encode([text for _, text in items])
            if encoded.shape != (len(items), 384):
                raise ValueError('encoder returned an unexpected embedding shape')
            start = 0 if cls._vectors is None else len(cls._vectors)
            cls._vectors = (encoded.copy() if cls._vectors is None
                            else np.concatenate([cls._vectors, encoded], axis=0))
            if cls._index is None:
                cls._index = {}
            for offset, (digest, _) in enumerate(items):
                cls._index[digest] = start + offset
            for position, digest in enumerate(hashes):
                cached = cls._index.get(digest)
                if cached is not None:
                    result[position] = cls._vectors[cached]
            self._save_cache()
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
    ENABLED = True
