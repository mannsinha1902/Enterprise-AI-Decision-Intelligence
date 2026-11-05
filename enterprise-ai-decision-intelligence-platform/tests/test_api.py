import json
import shutil
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.features import FEATURES
from src.torch_model import load_torch_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
VALID_FEATURES = {
    "recency_days": 45,
    "orders_90d": 3,
    "total_spend": 250.0,
    "support_tickets": 1,
}


@pytest.fixture
def api_client():
    with TestClient(create_app(ARTIFACT_DIR)) as client:
        yield client


def test_health_and_models_report_saved_artifacts(api_client):
    health = api_client.get("/health")
    models = api_client.get("/models")
    metadata = json.loads(
        (ARTIFACT_DIR / "model_metadata.json").read_text(encoding="utf-8")
    )

    assert health.status_code == 200
    assert health.json() == {
        "status": "ready",
        "model_loading_status": {
            "logistic_regression": "loaded",
            "xgboost": "loaded",
            "pytorch": "loaded",
        },
        "available_models": ["logistic_regression", "xgboost", "pytorch"],
    }
    assert models.json() == {
        "models": [
            {"name": name, "version": metadata["model_version"]}
            for name in ("logistic_regression", "xgboost", "pytorch")
        ]
    }


@pytest.mark.parametrize(
    ("model_name", "filename", "loader"),
    [
        ("logistic_regression", "churn_model.joblib", joblib.load),
        ("xgboost", "xgboost_model.joblib", joblib.load),
        ("pytorch", "pytorch_model.pt", load_torch_model),
    ],
)
def test_predict_uses_each_saved_model(model_name, filename, loader, api_client):
    features = pd.DataFrame([VALID_FEATURES], columns=FEATURES)
    model = loader(ARTIFACT_DIR / filename)
    expected_probability = float(model.predict_proba(features)[0, 1])

    response = api_client.post(
        "/predict",
        json={"model_name": model_name, "features": VALID_FEATURES},
    )

    assert response.status_code == 200
    assert response.json()["model_name"] == model_name
    assert response.json()["model_version"]
    assert response.json()["prediction"] == (
        "churn" if expected_probability >= 0.5 else "no_churn"
    )
    assert response.json()["churn_probability"] == pytest.approx(
        expected_probability
    )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "features": {
                **VALID_FEATURES,
                "recency_days": -1,
            }
        },
        {"features": {"recency_days": 45, "orders_90d": 3, "total_spend": 250}},
        {"features": {**VALID_FEATURES, "unexpected": 1}},
        {"features": VALID_FEATURES, "model_name": "unknown"},
        {"features": {**VALID_FEATURES, "support_tickets": 1.5}},
    ],
)
def test_predict_rejects_invalid_inputs(api_client, payload):
    response = api_client.post("/predict", json=payload)

    assert response.status_code == 422


def test_missing_artifacts_report_not_ready_and_return_service_error(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        health = client.get("/health")
        models = client.get("/models")
        prediction = client.post(
            "/predict",
            json={"model_name": "logistic_regression", "features": VALID_FEATURES},
        )

    assert health.json()["status"] == "not_ready"
    assert health.json()["available_models"] == []
    assert models.json() == {"models": []}
    assert prediction.status_code == 503
    assert prediction.json()["detail"] == (
        "No trained models are available. Run train.py first."
    )


def test_model_selection_reports_unavailable_saved_model(tmp_path):
    shutil.copy2(
        ARTIFACT_DIR / "churn_model.joblib",
        tmp_path / "churn_model.joblib",
    )
    with TestClient(create_app(tmp_path)) as client:
        response = client.post(
            "/predict",
            json={"model_name": "pytorch", "features": VALID_FEATURES},
        )

    assert response.status_code == 503
    assert "pytorch" in response.json()["detail"]
