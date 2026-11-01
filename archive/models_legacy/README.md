# Archived model library

This is the pre-cleanup `models` package, kept as a development traceback.

It contains the V2/V3 inheritance chain, comparable/spline/text feature
experiments, building models, XGBoost variants, residual correction, and neural
network experiments. Internal imports still use the absolute `models.*` paths,
so this directory is **reference only** and is not importable as `models`
without rewriting those imports.

The active clean package is the sibling `models/` directory at the repository
root. It contains only `BaseModel.py`, `features.py`, `final.py` and
`__init__.py`, with four model classes: off-market and list-aware versions of
the advanced feature model in both HGBR and XGBoost.
