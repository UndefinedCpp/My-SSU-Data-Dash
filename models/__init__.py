# Workaround for macOS OpenMP runtime conflict between PyTorch and XGBoost
try:
    import xgboost as _xgboost  # noqa: F401
except Exception:
    _xgboost = None
