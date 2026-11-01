"""Clean, self-contained final model package.

Only the advanced feature pipeline and its two learners are kept:

* ``OffMarketBuildingV1`` — HGBR
* ``OffMarketBuildingXGBoostV1`` — XGBoost
* ``ListAwareBuildingV1`` — HGBR
* ``ListAwareBuildingXGBoostV1`` — XGBoost

Feature construction lives in ``features.py``. The archived development
library is at ``archive/models_legacy`` and is not imported by this package.
"""
