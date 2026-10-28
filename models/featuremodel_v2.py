from models.featuremodel import *

HOUSE_STYLES = [
    'Apartment',
    'Bungalow', 
    '2-Storey',
    '3-Storey',
    '2 1/2 Storey',
    'Bungalow-Raised',
    'Loft',
    'Stacked Townhse',
    'Backsplit 4',
    'Sidesplit 3',
    'Multi-Level',
    'Backsplit 3',
    '1 1/2 Storey',
    'Sidesplit 4',
    '1 Storey/Apt',
    'Backsplit 5'
]
MUNICIPALITIES = ['Toronto', 'North York', 'Scarborough', 'Etobicoke', 'York']

class OffMarketFeatureModelV2(SimpleOffMarketFeatureModel):
    """
    Off-market model version 2 with additional features for improved prediction
    accuracy.
    """

    NAME = "Off-Market Feature Model V2"
    ENABLED = True
    MODEL_TAG = "model"

    def __init__(self):
        super().__init__()
        self.model = HistGradientBoostingRegressor(
            categorical_features=['community']
        )

    @staticmethod
    def make_features(homes: pd.DataFrame) -> pd.DataFrame:
        X = SimpleOffMarketFeatureModel.make_features(homes)
        # Add extra features
        X['bed_per_bath'] = homes['bedrooms'] / homes['washrooms'].replace(0, 1)
        X['lot_area'] = homes['lot_front'] * homes['lot_depth']

        lon, lat = homes['longitude'], homes['latitude']
        dx = (lon - (-79.38)) * 111 * 0.72
        dy = (lat - 43.72) * 111
        X['distance_center'] = np.sqrt(dx ** 2 + dy ** 2)

        house_style = homes['house_style']
        for style in HOUSE_STYLES:
            X["house_style:" + style] = (house_style == style).astype(int)

        municipality = homes['municipality']
        for muni in MUNICIPALITIES:
            X["municipality:" + muni] = (municipality == muni).astype(int)

        X["community"] = homes["community"]
        
        X["month_of_year"] = homes["date_listed"].dt.month

        return X


class ListAwareFeatureModelV2(SimpleListAwareFeatureModel):

    NAME = "List-Aware Feature Model V2"
    ENABLED = True
    MODEL_TAG = "model"

    def __init__(self):
        super().__init__()
        self.model = HistGradientBoostingRegressor(
            categorical_features=["community"]
        )

    @staticmethod
    def make_list_aware_features(homes: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame()
        X["list_price"] = homes["list_price"]
        X["log_list_price"] = np.log(homes["list_price"])
        X["list_price_per_sqft"] = homes["list_price"] / (homes["lot_front"] * homes["lot_depth"])
        return X

    @staticmethod
    def make_features(homes: pd.DataFrame) -> pd.DataFrame:
        X = OffMarketFeatureModelV2.make_features(homes)
        list_aware_features = ListAwareFeatureModelV2.make_list_aware_features(homes)
        X = pd.concat([X, list_aware_features], axis=1)
        return X