"""Preserve the original native CatBoost bytes across repeated pickle saves.

The pinned CatBoost text learner can corrupt its dictionary when serializing a
previously deserialized model. Reusing its original native state avoids that
second serialization. A refit invalidates the saved state.
"""
import pickle
from catboost import CatBoostRegressor


class StableCatBoostRegressor(CatBoostRegressor):
    def __getstate__(self):
        state = getattr(self, '_stable_pickle_state', None)
        if state is None:
            state = super().__getstate__()
            self._stable_pickle_state = dict(state)
        return dict(state)

    def __setstate__(self, state):
        self._stable_pickle_state = dict(state)
        # CatBoost's implementation deletes entries from the argument.
        super().__setstate__(dict(state))

    def fit(self, *args, **kwargs):
        self._stable_pickle_state = None
        return super().fit(*args, **kwargs)

    def set_params(self, **kwargs):
        self._stable_pickle_state = None
        return super().set_params(**kwargs)

    def load_model(self, *args, **kwargs):
        self._stable_pickle_state = None
        return super().load_model(*args, **kwargs)

    def shrink(self, *args, **kwargs):
        self._stable_pickle_state = None
        return super().shrink(*args, **kwargs)


class _ModelUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'catboost.core' and name == 'CatBoostRegressor':
            return StableCatBoostRegressor
        return super().find_class(module, name)


def load_fitted_model(handle):
    """Read our fitted artifacts, upgrading legacy CatBoost regressors."""
    return _ModelUnpickler(handle).load()
