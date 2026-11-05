# Archived model library

This directory preserves model development history. Nothing here is scanned by
model discovery or imported by the submission defaults.

## Unused standalone variants

[unused_variants.py](unused_variants.py) preserves four implementations removed
from the active package during the selected-model cleanup:

- `OffMarketBuildingV1` — off-market HGB building model.
- `ListAwareBuildingXGBoostV1` — list-aware XGBoost building model.
- `ListAwareValueGapV5` — list-aware CatBoost value-gap model.
- `OffMarketRobustHGBV6` — off-market absolute-error HGB model.

These classes are reference code, not selectable backtest or benchmark models.
They reuse shared helpers from the active package. Their old `models.*` class
paths are not compatibility aliases; use the original Git revision when opening
checkpoints fitted with these archived classes.

`ListAwareValueGapHGBV5` and `OffMarketRobustXGBV6` remain in the active package
because the selected text and semantic models inherit their implementation.
They are disabled as standalone candidates; their selected subclasses are
explicitly enabled.

## Earlier experiments

The remaining files are the older pre-cleanup model library: V2/V3 inheritance
chains, comparable/spline/text features, building models, XGBoost variants,
residual corrections, and neural-network experiments. Their internal imports
still use the original absolute `models.*` paths, so they are **reference only**
and are not a standalone importable copy of the old package.

See the [root README](../../README.md#active-models) for the active dependency
tree and selected model weights.
