"""
Baseline models that ships with the starter kit.

Original script: <https://github.com/erwinzhang7/SSU-Data-Dash/blob/main/models.py>
"""

from models.BaseModel import BaseModel
import numpy as np
import pandas as pd

class ListPriceBaseline(BaseModel):
    """List-aware baseline: every home sells for exactly its list price."""
    NAME = 'Copy List Price'
    TRACKS = ('list_aware',)
    ENABLED = True
    MODEL_TAG = 'baseline'

    def train(self, train):
        return self   # nothing to learn

    def predict(self, homes):
        return pd.Series(homes["list_price"].to_numpy(dtype=float))


class NearestSalesBaseline(BaseModel):
    """Off-market baseline: the median sold price of the k nearest earlier sales of
    the same property type.

    "Nearest" means closest by latitude and longitude. "Earlier" means sold before the
    home was listed, so the model never looks into the future, even when the
    training table holds later sales. It makes no adjustment for size, condition or
    for prices moving since those sales: improving on that is your job.
    """
    NAME = 'Nearest Sales'
    TRACKS = ('off_market',)
    ENABLED = True
    MODEL_TAG = 'baseline'

    def __init__(self, k=10):
        self.k = k

    def train(self, train):
        columns = ["ml_num", "house_type_name", "date_sold", "latitude", "longitude", "sold_price"]
        # Newest sales first, so when several sales are equally near (a condo
        # building shares one location), the most recent ones are picked.
        self.sales = train[columns].sort_values(["date_sold", "ml_num"], ascending=[False, True])
        self.sales_by_type = dict(list(self.sales.groupby("house_type_name")))
        return self

    def predict(self, homes):
        x = np.array([self.predict_one(home) for _, home in homes.iterrows()])
        return pd.Series(x)

    def predict_one(self, home):
        sales = self.sales_by_type.get(home["house_type_name"], self.sales)
        earlier = sales[sales["date_sold"] < home["date_listed"]]
        if len(earlier) == 0:   # no earlier sale of this type: use every type
            earlier = self.sales[self.sales["date_sold"] < home["date_listed"]]

        # Distance in degrees. In Toronto a degree of longitude is only 0.72 times as
        # long as a degree of latitude (0.72 = cosine of 43.7 degrees north).
        dx = (earlier["longitude"].to_numpy() - home["longitude"]) * 0.72
        dy = earlier["latitude"].to_numpy() - home["latitude"]
        distance = np.sqrt(dx ** 2 + dy ** 2)

        nearest = np.argsort(distance, kind="stable")[: self.k]
        return float(np.median(earlier["sold_price"].to_numpy()[nearest]))


class RandomBaseline(BaseModel):
    NAME = 'Random Baseline'
    TRACKS = ('list_aware',)
    ENABLED = True
    MODEL_TAG = 'baseline'

    def train(self, train):
        return self   # nothing to learn

    def predict(self, homes):
        return pd.Series(np.random.rand(len(homes)) * homes["list_price"].to_numpy(dtype=float))