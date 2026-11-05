"""Submission models and their required components.

Defaults: ``OffMarketRefinedV6`` and ``ListAwareRefinedV6``.

The off-market model blends the original building XGBoost model, categorical/
text CatBoost, and semantic-feature XGBoost. The list-aware model blends the
original building HGB model with the historical text stack. Blend settings
live in ``refined_config.json``.

Only these defaults and their selected components are enabled for discovery.
The HGB value-gap and robust XGBoost bases are retained for inheritance, not
as standalone benchmark candidates. Unused standalone variants are preserved
in ``archive/models_legacy/unused_variants.py`` and are never imported here.
"""
