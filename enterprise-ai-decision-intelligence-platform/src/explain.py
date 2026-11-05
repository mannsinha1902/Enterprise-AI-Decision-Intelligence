import shap

from src.ml import FEATURES


def explain_xgboost(model, data):
    """Return per-feature SHAP contributions for the XGBoost model's raw scores."""
    return shap.TreeExplainer(model)(data[FEATURES], check_additivity=True)
