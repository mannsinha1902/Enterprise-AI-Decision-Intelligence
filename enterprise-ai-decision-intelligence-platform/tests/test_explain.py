import numpy as np

from src.explain import explain_xgboost
from src.ml import FEATURES, compare_models, make_data


def test_xgboost_shap_contributions_sum_to_raw_model_score():
    data = make_data(300)
    models, _ = compare_models(data)
    features = data[FEATURES].head(12)

    explanation = explain_xgboost(models["xgboost"], features)
    reconstructed_scores = explanation.base_values + explanation.values.sum(axis=1)
    raw_scores = models["xgboost"].predict(features, output_margin=True)

    assert explanation.values.shape == (12, len(FEATURES))
    assert explanation.feature_names == FEATURES
    np.testing.assert_allclose(reconstructed_scores, raw_scores, rtol=1e-4, atol=1e-4)
