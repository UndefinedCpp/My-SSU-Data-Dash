"""Frozen development-selected blends; settings live in refined_config.json."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from models.BaseModel import BaseListAwareModel, BaseOffMarketModel


class _RefinedBlend:
    ENABLED = True
    MODEL_TAG = 'Refined'

    def __init__(self):
        from model_hub import load_model
        config = json.loads(Path(__file__).with_name('refined_config.json').read_text())[self.TRACK]
        self.weight = config['candidate_weight']
        self.incumbent = load_model(config['incumbent'])
        self.candidate = load_model(config['candidate']) if self.weight else None

    def _train(self, data):
        self.incumbent.train(data)
        if self.candidate is not None:
            self.candidate.train(data)
        return self

    def _predict(self, homes):
        if homes.empty:
            return pd.Series(index=homes.index, dtype=float)
        prediction = np.log(np.asarray(self.incumbent.predict(homes)))
        if self.candidate is not None:
            prediction = ((1 - self.weight) * prediction + self.weight *
                          np.log(np.asarray(self.candidate.predict(homes))))
        return pd.Series(np.exp(prediction), index=homes.index)


class OffMarketRefinedV6(_RefinedBlend, BaseOffMarketModel):
    NAME = 'Off-Market Refined V6'


class ListAwareRefinedV6(_RefinedBlend, BaseListAwareModel):
    NAME = 'List-Aware Refined V6'

