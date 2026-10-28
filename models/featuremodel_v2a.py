from models.featuremodel_v2 import *
import unicodedata

TEXT_CUE_WORDBANK = {
    'text:renovated': ['renovated', 'renovation', 'updated', 'upgraded',
                       'refreshed', 'redone'],
    'text:original': ['original', 'dated', 'as-is', 'worn', 'tired',
                      'outdated'],
    'text:potential': ['potential', 'investor'],
    'text:luxury': ['luxury', 'luxurious', 'stunning', 'high-end',
                    'premium'],
    'text:view': ['view', 'skyline', 'ravine', 'waterfront']
}

class OffMarketFeatureModelV2a(OffMarketFeatureModelV2):
    
    NAME = "Off-Market Feature Model V2a (w/ text cues)"
    ENABLED = True
    MODEL_TAG = "model"

    @staticmethod
    def make_features(homes: pd.DataFrame) -> pd.DataFrame:
        X = OffMarketFeatureModelV2.make_features(homes)

        def normalize_string(s: str) -> str:
            s = s.strip().lower()
            s = unicodedata.normalize('NFKD', s)
            return s
        def count_occurrences(s: str, keywords: list) -> int:
            return sum(s.count(k) for k in keywords)
        for cue, keywords in TEXT_CUE_WORDBANK.items():
            normalized_remark = homes['remarks'].apply(normalize_string)
            X[cue] = normalized_remark.apply(lambda r: count_occurrences(r, keywords))
        return X

class ListAwareFeatureModelV2a(ListAwareFeatureModelV2):

    NAME = "List-Aware Feature Model V2a (w/ text cues)"
    ENABLED = True
    MODEL_TAG = "model"

    @staticmethod
    def make_features(homes: pd.DataFrame) -> pd.DataFrame:
        X = OffMarketFeatureModelV2a.make_features(homes)
        list_aware_features = ListAwareFeatureModelV2a.make_list_aware_features(homes)
        X = pd.concat([X, list_aware_features], axis=1)
        return X