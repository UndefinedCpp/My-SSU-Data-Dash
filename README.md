# SSU Data Dash: predict Toronto home prices

Competition repo: [erwinzhang7/SSU-Data-Dash](https://github.com/erwinzhang7/SSU-Data-Dash/)

Some code are derived from the original repo.

## Setup

Python 3.13.13

```bash
pip install -r requirements.txt
```

## Commands

1. `python run.py predict` **(Required)** Predict using the final, trained model(s) and generate two submission files.
2. `python run.py check` **(Required)** Check the two submission files for format and consistency.
3. `python run.py backtest [model_name] [-t track] [-c cutoff]` Backtest the specified model on the given track and cutoff date. Set cutoff to `all` to initiate a walk-forward backtest from Dec. 2025 to May 2026.
4. `python run.py clean` Clean up generated backtest files.
5. `python run.py benchmark [-t track]` (under construction)