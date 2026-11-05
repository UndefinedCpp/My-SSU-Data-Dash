"""Categorical and text boosting with strictly historical sale features."""
import numpy as np
import pandas as pd
from models.serialization import StableCatBoostRegressor as CatBoostRegressor

from models.BaseModel import BaseOffMarketModel
from models.features import BuildingBlock, ComparableBlock, RemarksTermBlock, base_features


class _CatBoostFeatures:
    LIST_AWARE = False

    def __init__(self, iterations=1200):
        self.iterations = iterations
        self.building = BuildingBlock(include_ratio=self.LIST_AWARE)
        self.comps = ComparableBlock()
        self.terms = RemarksTermBlock()
        self.model = None

    def _features(self, homes, fit=False):
        frame = base_features(homes, list_aware=self.LIST_AWARE)
        frame['garage'] = homes['garage']
        frame['lot_front'] = homes['lot_front']
        frame['lot_depth'] = homes['lot_depth']
        frame['log_area'] = np.log(frame['area'].clip(lower=1))
        frame['area_per_bedroom'] = frame['area'] / homes['bedrooms'].clip(lower=1)
        frame['lot_to_floor_ratio'] = frame['lot_area'] / frame['area'].clip(lower=1)
        for column in ('house_type_name', 'house_style', 'municipality', 'community'):
            frame[column] = homes[column].fillna('__missing__').astype(str)
        frame['community_type'] = frame['community'] + '|' + frame['house_type_name']
        # The text vocabulary is fixed, so neither test labels nor test text are fitted.
        text = self.terms.fit_transform(homes) if fit else self.terms.transform(homes)
        building = self.building.fit_transform(homes) if fit else self.building.transform(homes)
        parts = [frame, text, building]
        if not self.LIST_AWARE:
            parts.append(self.comps.fit_transform(homes) if fit else self.comps.transform(homes))
        frame = pd.concat(parts, axis=1)
        # CatBoost learns condition/amenity phrases beyond the small fixed wordbank.
        frame['remarks_text'] = homes['remarks'].fillna('').astype(str).str.strip()
        frame.loc[frame['remarks_text'].eq(''), 'remarks_text'] = 'missing remarks'
        numeric = frame.select_dtypes(include='number').columns
        frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan).fillna(-999)
        return frame

    def _train(self, data):
        frame = self._features(data, fit=True)
        target = np.log(data['sold_price'].to_numpy(float))
        if self.LIST_AWARE:
            target -= np.log(data['list_price'].to_numpy(float))
        self.model = CatBoostRegressor(
            iterations=self.iterations, depth=7, learning_rate=0.05,
            loss_function='MAE' if self.LIST_AWARE else 'RMSE',
            l2_leaf_reg=5, random_seed=42, thread_count=8,
            cat_features=['house_type_name', 'house_style', 'municipality',
                          'community', 'community_type'],
            text_features=['remarks_text'],
            feature_calcers=['BoW:top_tokens_count=1500'],
            allow_writing_files=False, verbose=False,
        )
        self.model.fit(frame, target)
        return self

    def _predict(self, homes):
        if self.model is None:
            raise RuntimeError('Call train before predict')
        if len(homes) == 0:
            return pd.Series(index=homes.index, dtype=float)
        prediction = self.model.predict(self._features(homes))
        if self.LIST_AWARE:
            prediction += np.log(homes['list_price'].to_numpy(float))
        return pd.Series(np.exp(np.clip(prediction, 5, 20)), index=homes.index)


class OffMarketCatBoostV2(_CatBoostFeatures, BaseOffMarketModel):
    NAME = 'Off-Market CatBoost V2'
    MODEL_TAG = 'CatBoost'
    ENABLED = True
