"""A temporally cross-fitted text signal for the sale-to-list ratio."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from models.hybrid_v5 import ListAwareValueGapHGBV5


class HistoricalTextSignal:
    def __init__(self, min_history=1000):
        self.min_history = min_history
        self.snapshots = {}

    def _fit_snapshot(self, history):
        if len(history) < self.min_history:
            return None
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=10,
                                     max_features=12000, strip_accents='unicode',
                                     sublinear_tf=True)
        matrix = vectorizer.fit_transform(history.remarks.fillna('').astype(str))
        model = Ridge(alpha=10, solver='lsqr', tol=1e-4)
        target = np.log(history.sold_price.to_numpy(float) / history.list_price.to_numpy(float))
        model.fit(matrix, target)
        return vectorizer, model

    @staticmethod
    def _cutoff(date):
        return pd.Timestamp(year=date.year, month=1 if date.month <= 6 else 7, day=1)

    def fit_transform(self, sales):
        self.snapshots = {}
        cutoffs = sales.date_listed.map(self._cutoff)
        values = np.zeros(len(sales), dtype=float)
        for cutoff in sorted(cutoffs.unique()):
            history = sales[sales.date_sold < cutoff]
            snapshot = self._fit_snapshot(history)
            self.snapshots[cutoff] = snapshot
            positions = np.flatnonzero(cutoffs.eq(cutoff).to_numpy())
            if snapshot is not None:
                vectorizer, model = snapshot
                text = sales.iloc[positions].remarks.fillna('').astype(str)
                values[positions] = model.predict(vectorizer.transform(text))
        final_cutoff = sales.date_sold.max().normalize() + pd.Timedelta(days=1)
        self.snapshots[final_cutoff] = self._fit_snapshot(sales)
        return pd.Series(values, index=sales.index)

    def transform(self, homes):
        if not self.snapshots:
            raise RuntimeError('Fit historical text signal before transform')
        values = np.zeros(len(homes), dtype=float)
        for month, positions in homes.groupby(homes.date_listed.dt.to_period('M')).indices.items():
            eligible = [cutoff for cutoff in self.snapshots if cutoff <= month.start_time]
            if not eligible:
                continue
            snapshot = self.snapshots[max(eligible)]
            if snapshot is not None:
                vectorizer, model = snapshot
                values[positions] = model.predict(vectorizer.transform(homes.iloc[positions].remarks.fillna('').astype(str)))
        return pd.Series(values, index=homes.index)


class ListAwareTextStackV6(ListAwareValueGapHGBV5):
    NAME = 'List-Aware Historical Text Stack V6'

    def __init__(self, iterations=1600):
        super().__init__(iterations=iterations)
        self.text_signal = HistoricalTextSignal()

    def _features(self, homes, fit=False):
        frame = super()._features(homes, fit=fit).copy()
        frame['historical_text_ratio'] = (self.text_signal.fit_transform(homes) if fit
                                          else self.text_signal.transform(homes))
        return frame
