"""Fast model-wiring regressions: no dataset, encoder download, or full training."""
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import pytest

import model_hub
from models.attempt1 import ListAwareBuildingV1, OffMarketBuildingXGBoostV1
from models.catboost_v2 import OffMarketCatBoostV2
from models.hybrid_v5 import ListAwareValueGapHGBV5
from models.robust_v6 import OffMarketRobustXGBV6
from models.semantic_v7 import FrozenSentenceEncoder, OffMarketSemanticV7
from models.text_stack_v6 import ListAwareTextStackV6

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_MODELS = {
    'OffMarketRefinedV6': 'off_market',
    'OffMarketEnsembleV2': 'off_market',
    'OffMarketBuildingXGBoostV1': 'off_market',
    'OffMarketCatBoostV2': 'off_market',
    'OffMarketSemanticV7': 'off_market',
    'ListAwareRefinedV6': 'list_aware',
    'ListAwareBuildingV1': 'list_aware',
    'ListAwareTextStackV6': 'list_aware',
}
ARCHIVED_MODELS = (
    'OffMarketBuildingV1',
    'ListAwareBuildingXGBoostV1',
    'ListAwareValueGapV5',
    'OffMarketRobustHGBV6',
)
IMPLEMENTATION_BASES = ('ListAwareValueGapHGBV5', 'OffMarketRobustXGBV6')


@pytest.fixture(autouse=True)
def repository_directory(monkeypatch):
    # The current model hub resolves its source directory from the working directory.
    monkeypatch.chdir(ROOT)


def test_discovery_contains_only_selected_models_and_components():
    discovered = model_hub.gather_models(ROOT / 'models')
    assert {cls.__name__: cls.TRACK for cls in discovered} == ACTIVE_MODELS


@pytest.mark.parametrize('selector,track', ACTIVE_MODELS.items())
def test_selected_models_load_and_keep_pickle_paths(selector, track):
    model = model_hub.load_model(selector)
    restored = pickle.loads(pickle.dumps(model))
    assert restored.get_id() == selector
    assert restored.TRACK == track
    assert restored.ENABLED is True
    assert type(restored).__module__.startswith('models.')


@pytest.mark.parametrize('selector', ARCHIVED_MODELS + IMPLEMENTATION_BASES)
def test_unused_variants_and_implementation_bases_are_not_selectable(selector):
    with pytest.raises(ValueError, match='not found'):
        model_hub.load_model(selector)


def test_selected_configuration_is_unchanged():
    config = json.loads((ROOT / 'models/refined_config.json').read_text())
    assert config == {
        'off_market': {
            'incumbent': 'OffMarketEnsembleV2',
            'candidate': 'OffMarketSemanticV7',
            'candidate_weight': 0.25,
        },
        'list_aware': {
            'incumbent': 'ListAwareBuildingV1',
            'candidate': 'ListAwareTextStackV6',
            'candidate_weight': 0.5,
        },
    }


@pytest.mark.parametrize('cls,list_aware', [
    (ListAwareTextStackV6, True),
    (OffMarketSemanticV7, False),
])
def test_selected_children_preserve_inherited_feature_settings(cls, list_aware):
    model = cls()
    assert model.iterations == 1600
    assert model.LIST_AWARE is list_aware
    assert model.building.include_ratio is list_aware
    assert model.ENABLED is True
    assert model.MODEL_TAG == ('HGBR' if list_aware else 'XGB')


@pytest.mark.parametrize('selector,track,leaf_prices,expected', [
    ('OffMarketRefinedV6', 'off_market', {
        OffMarketBuildingXGBoostV1: 100.0,
        OffMarketCatBoostV2: 400.0,
        OffMarketSemanticV7: 1600.0,
    }, 100.0 ** 0.375 * 400.0 ** 0.375 * 1600.0 ** 0.25),
    ('ListAwareRefinedV6', 'list_aware', {
        ListAwareBuildingV1: 100.0,
        ListAwareTextStackV6: 400.0,
    }, 200.0),
])
def test_default_training_dispatch_and_geometric_blends(
    monkeypatch, selector, track, leaf_prices, expected,
):
    trained = []
    predicted = []

    def assert_track_input(frame):
        assert ('list_price' in frame.columns) == (track == 'list_aware')

    def train_leaf(self, frame):
        assert_track_input(frame)
        trained.append(type(self))
        return self

    def predict_leaf(self, frame):
        assert_track_input(frame)
        predicted.append(type(self))
        return pd.Series(leaf_prices[type(self)], index=frame.index)

    for cls in leaf_prices:
        monkeypatch.setattr(cls, '_train', train_leaf)
        monkeypatch.setattr(cls, '_predict', predict_leaf)

    homes = pd.DataFrame({
        'list_price': [123.0, 456.0],
        'remarks': ['first home', 'second home'],
    }, index=[9, 4])
    model = model_hub.load_model(selector)
    model.train(homes.assign(sold_price=[150.0, 500.0]))
    prediction = model.predict(homes)
    assert trained == list(leaf_prices)
    assert predicted == list(leaf_prices)
    assert prediction.index.equals(homes.index)
    np.testing.assert_allclose(prediction, expected)


@pytest.mark.parametrize('selector', ['OffMarketRefinedV6', 'ListAwareRefinedV6'])
def test_defaults_preserve_empty_prediction_behavior(selector):
    prediction = model_hub.load_model(selector).predict(pd.DataFrame({'remarks': []}))
    assert prediction.empty


@pytest.mark.parametrize('cls,module,expected', [
    (OffMarketRobustXGBV6, 'models.robust_v6', {
        'objective': 'reg:absoluteerror', 'n_estimators': 1600,
        'max_depth': 5, 'learning_rate': 0.035, 'min_child_weight': 30,
        'subsample': 0.85, 'colsample_bytree': 0.85, 'reg_lambda': 8,
        'tree_method': 'hist', 'enable_categorical': True, 'n_jobs': 8,
        'random_state': 44, 'verbosity': 0,
    }),
    (ListAwareValueGapHGBV5, 'models.hybrid_v5', {
        'loss': 'absolute_error', 'learning_rate': 0.04, 'max_iter': 650,
        'max_leaf_nodes': 31, 'min_samples_leaf': 40, 'l2_regularization': 5,
        'early_stopping': False, 'categorical_features': ['community'],
        'random_state': 43,
    }),
])
def test_reparented_models_preserve_training_parameters(monkeypatch, cls, module, expected):
    from importlib import import_module
    from unittest.mock import Mock

    model = cls()
    frame = pd.DataFrame({'community': ['A', 'B']})
    monkeypatch.setattr(model, '_features', lambda *args, **kwargs: frame)
    monkeypatch.setattr(model, '_numeric', lambda *args, **kwargs: frame)
    learner = Mock()
    attribute = 'XGBRegressor' if cls is OffMarketRobustXGBV6 else 'HistGradientBoostingRegressor'
    monkeypatch.setattr(import_module(module), attribute, learner)
    model.train(pd.DataFrame({'sold_price': [100.0, 200.0], 'list_price': [90.0, 180.0]}))
    assert learner.call_args.kwargs == expected


def test_model_discovery_scans_only_the_active_package(monkeypatch):
    requested_modules = []
    import_module = model_hub._import_module

    def record_import(module_name):
        requested_modules.append(module_name)
        return import_module(module_name)

    monkeypatch.setattr(model_hub, '_import_module', record_import)
    model_hub.gather_models(ROOT / 'models')
    assert requested_modules
    assert all(name.startswith('models.') for name in requested_modules)


class _FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), dict(kwargs)))
        return np.stack([np.full(384, len(text), dtype=np.float32) for text in texts])


@pytest.fixture
def embedding_cache(tmp_path):
    FrozenSentenceEncoder._reset()
    yield tmp_path
    FrozenSentenceEncoder._reset()


def test_embeddings_are_encoded_once_per_distinct_text(embedding_cache):
    fake = _FakeEncoder()
    FrozenSentenceEncoder._encoder = fake
    frame = pd.DataFrame({'remarks': ['a', 'bb', 'a']})
    vectors = FrozenSentenceEncoder(cache_dir=embedding_cache).transform(frame)

    assert fake.calls[0][0] == ['a', 'bb']
    assert fake.calls[0][1]['convert_to_tensor'] is True
    assert np.array_equal(vectors[0], vectors[2])
    assert not np.array_equal(vectors[0], vectors[1])
    assert {path.name for path in embedding_cache.iterdir()} == {
        'metadata.json', 'hashes.npy', 'vectors.npy',
    }


def test_embedding_cache_is_reused_across_encoder_instances(embedding_cache):
    first_fake = _FakeEncoder()
    FrozenSentenceEncoder._encoder = first_fake
    first = FrozenSentenceEncoder(cache_dir=embedding_cache).transform(
        pd.DataFrame({'remarks': ['a', 'bb']})
    )

    FrozenSentenceEncoder._reset()
    second_fake = _FakeEncoder()
    FrozenSentenceEncoder._encoder = second_fake
    second = FrozenSentenceEncoder(cache_dir=embedding_cache).transform(
        pd.DataFrame({'remarks': ['bb', 'a']})
    )

    assert second_fake.calls == []
    assert np.array_equal(second[0], first[1])
    assert np.array_equal(second[1], first[0])


def test_embedding_cache_rejects_stale_model_revision(embedding_cache):
    FrozenSentenceEncoder._encoder = _FakeEncoder()
    encoder = FrozenSentenceEncoder(cache_dir=embedding_cache)
    original = encoder.transform(pd.DataFrame({'remarks': ['a']}))

    metadata_path = embedding_cache / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['revision'] = 'stale'
    metadata_path.write_text(json.dumps(metadata))

    FrozenSentenceEncoder._reset()
    fresh_fake = _FakeEncoder()
    FrozenSentenceEncoder._encoder = fresh_fake
    refreshed = FrozenSentenceEncoder(cache_dir=embedding_cache).transform(
        pd.DataFrame({'remarks': ['a']})
    )

    assert len(fresh_fake.calls) == 1
    assert np.array_equal(refreshed, original)


def test_empty_remarks_do_not_load_the_encoder(embedding_cache):
    fake = _FakeEncoder()
    FrozenSentenceEncoder._encoder = fake
    vectors = FrozenSentenceEncoder(cache_dir=embedding_cache).transform(
        pd.DataFrame({'remarks': []})
    )
    assert vectors.shape == (0, 384)
    assert fake.calls == []
