# SSU Data Dash: predict Toronto home prices

Competition repo: [erwinzhang7/SSU-Data-Dash](https://github.com/erwinzhang7/SSU-Data-Dash/); Changelog [here](https://docs.google.com/spreadsheets/d/1rNc3G17JMyiek8kcmEtOW_g_7IQfK2x3/edit?gid=1719102416#gid=1719102416)

Some code are derived from the original repo.

## Setup

Python 3.13.13

```bash
pip install -r requirements.txt
```

## Commands

1. `python run.py predict [--save-model DIR]` **(Required)** Predict using the final, trained model(s) and generate two submission files. Optionally save the two final fitted models and metadata.
2. `python run.py check` **(Required)** Check the two submission files for format and consistency.
3. `python run.py backtest [model_name] [-c cutoff] [--ref REF] [--cohort COHORT]` Backtest the specified model on its own track (its `TRACK`) and the given cutoff date. Cutoff defaults to `all`, which runs a walk-forward backtest from Dec. 2025 to May 2026. `--cohort` defaults to `date_listed`.
4. `python run.py fulltest [model_name] [--ref REF] [--cohort COHORT] [--save-model DIR]` Train through January 2026 and evaluate February–May 2026. `--cohort` defaults to `date_listed`.
5. `python run.py clean` Clean up generated backtest files.
6. `python run.py benchmark [-t track]` (under construction)

## Evaluation cohorts

`--cohort date_listed` is the default. Training uses sales completed before
the cutoff, while validation uses homes listed on or after the cutoff (or in
the cutoff month for walk-forward backtests). This avoids showing the model
sales that happened after a validation home was listed.

Use `--cohort date_sold` only to reproduce the older protocol, which selects
validation homes by sale date and can include homes listed before the training
cutoff.

## Creating custom models

1. Add a `.py` file under `models/` (any name except `BaseModel.py`).
2. Inherit the base class for your track, from `models.BaseModel`:
   - `BaseOffMarketModel`: homes with no list price (`list_price` is removed from the data automatically).
   - `BaseListAwareModel`: homes with a `list_price`.
3. Set `NAME`, `MODEL_TAG` and `ENABLED = True` (models are disabled by default and ignored by `model_hub`).
4. Implement `__init__` (optionally), `_train(self, data)`, which fits the model on past sales (`data` includes `sold_price`), and `_predict(self, data)`, which returns a `pd.Series` or a `np.ndarray`.

Example:

```python
import pandas as pd
from models.BaseModel import BaseOffMarketModel

class MedianPrice(BaseOffMarketModel):
    NAME = 'Median Price'
    MODEL_TAG = 'example'
    ENABLED = True

    def _train(self, data):
        self.median = data['sold_price'].median()

    def _predict(self, data):
        return pd.Series(self.median, index=range(len(data)))
```

Then run `python run.py backtest MedianPrice`. The model is selected by its class name. To use it for the final submission, set it in `SUBMISSION_MODELS` in `run.py`.

A model works on one track only. To cover both tracks, write two classes. 
