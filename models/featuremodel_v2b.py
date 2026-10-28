from models.featuremodel_v2a import *

class OffMarketFeatureModelV2b(OffMarketFeatureModelV2a):

    NAME = "Off-Market Feature Model V2b (more trees)"
    ENABLED = True
    MODEL_TAG = "model"

    def __init__(self):
        super().__init__()
        self.model = HistGradientBoostingRegressor(
            categorical_features=['community'],
            max_iter=200,
        )


class ListAwareFeatureModelV2b(ListAwareFeatureModelV2a):

    NAME = "List-Aware Feature Model V2b (more trees)"
    ENABLED = True
    MODEL_TAG = "model"

    def __init__(self):
        super().__init__()
        self.model = HistGradientBoostingRegressor(
            categorical_features=['community'],
            max_iter=200,
        )
