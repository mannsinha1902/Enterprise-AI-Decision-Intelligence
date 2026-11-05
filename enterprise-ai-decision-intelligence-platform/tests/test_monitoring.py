import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.database.operations import load_synthetic_customers
from src.database.session import create_database_engine, create_session_factory
from src.features import FEATURES
from src.monitoring import (
    MonitoringThresholds,
    build_monitoring_report,
    feature_drift_report,
    performance_degradation_report,
    prediction_drift_report,
)
from src.monitoring_cli import log_monitoring_report, run_monitoring

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


@pytest.fixture
def monitoring_database(tmp_path, monkeypatch):
    database_url = f"sqlite:///{(tmp_path / 'monitoring.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    command.upgrade(config, "head")
    engine = create_database_engine(database_url)
    factory = create_session_factory(engine)
    load_synthetic_customers(factory, count=100)
    yield database_url, factory
    engine.dispose()


def test_numeric_and_categorical_drift_alerts():
    rng = np.random.default_rng(42)
    reference = pd.DataFrame(
        {
            "amount": rng.normal(100, 5, 500),
            "channel": ["web"] * 400 + ["store"] * 100,
        }
    )
    current = pd.DataFrame(
        {
            "amount": rng.normal(175, 5, 500),
            "channel": ["store"] * 500,
        }
    )

    report = feature_drift_report(
        reference,
        current,
        MonitoringThresholds(min_samples=30),
    )

    assert report["status"] == "critical"
    numeric, categorical = report["features"]
    assert numeric["statistic"] == "psi"
    assert numeric["ks_p_value"] < 0.05
    assert numeric["status"] == "critical"
    assert categorical["statistic"] == "total_variation_distance"
    assert categorical["chi_square_p_value"] < 0.05
    assert categorical["status"] == "critical"


def test_drift_thresholds_and_small_samples_are_explicit():
    reference = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]})
    current = pd.DataFrame({"value": [100.0, 200.0, 300.0, 400.0]})

    report = feature_drift_report(
        reference,
        current,
        MonitoringThresholds(min_samples=30),
    )

    assert report["status"] == "insufficient_data"
    assert report["features"][0]["status"] == "insufficient_data"


def test_empty_prediction_batch_is_reported_without_failing():
    report = prediction_drift_report(
        [0.2, 0.3, 0.4],
        [],
        MonitoringThresholds(min_samples=2),
    )

    assert report["status"] == "insufficient_data"
    assert report["probability"]["value"] is None
    assert report["class_distribution"]["value"] is None


def test_labeled_batch_performance_degradation_is_separate_and_thresholded():
    labels = [0, 1] * 100
    reversed_scores = [0.9 if label == 0 else 0.1 for label in labels]
    report = performance_degradation_report(
        labels,
        reversed_scores,
        {"roc_auc": 0.99, "average_precision": 0.99, "f1": 0.99},
        MonitoringThresholds(
            min_samples=30,
            performance_warning_drop=0.1,
            performance_critical_drop=0.2,
        ),
    )

    assert report["status"] == "critical"
    assert report["current_metrics"]["roc_auc"] == 0
    assert report["metric_drops"]["f1"] >= 0.2
    assert "not independently verified" in report["basis"]


def test_prediction_report_separates_drift_from_performance():
    reference = np.linspace(0.05, 0.45, 500)
    current = np.linspace(0.55, 0.95, 500)
    report = prediction_drift_report(
        reference,
        current,
        MonitoringThresholds(min_samples=30),
    )
    combined = build_monitoring_report(
        pd.DataFrame({"feature": range(500)}),
        pd.DataFrame({"feature": range(500)}),
        {"model": reference.tolist()},
        {"model": current.tolist()},
        {"model": {"roc_auc": 0.8}},
        MonitoringThresholds(min_samples=30),
    )

    assert report["status"] == "critical"
    assert report["reference_churn_rate"] == 0
    assert report["current_churn_rate"] == 1
    assert combined["data_drift"]["status"] == "ok"
    assert combined["prediction_drift"]["model"]["status"] == "critical"
    assert combined["performance_degradation"]["status"] == "not_measured"
    assert "not production" in combined["holdout_metrics"]["model"]["source"]


def test_monitoring_script_detects_deliberately_shifted_synthetic_batch():
    report = run_monitoring(
        ARTIFACT_DIR / "synthetic_customers.csv",
        batch_size=500,
        seed=2026,
        shift_feature="recency_days",
        shift_amount=60,
        thresholds=MonitoringThresholds(min_samples=30),
    )

    recency = next(
        item
        for item in report["data_drift"]["features"]
        if item["feature"] == "recency_days"
    )
    assert report["periods"]["current"] == "synthetic batch (seed=2026)"
    assert report["prediction_drift"]["logistic_regression"]["reference_rows"] == 2000
    assert report["prediction_drift"]["logistic_regression"]["current_rows"] == 500
    assert recency["status"] in {"warning", "critical"}
    assert report["performance_degradation"]["status"] == "not_measured"


def test_monitoring_report_is_logged_to_existing_mlflow_experiment(tmp_path):
    report = {
        "data_drift": {
            "reference_rows": 100,
            "current_rows": 50,
            "status": "warning",
            "features": [{"feature": "recency_days", "value": 0.15}],
        },
        "prediction_drift": {
            "logistic_regression": {
                "probability": {"value": 0.12},
                "class_distribution": {"value": 0.2},
            }
        },
        "performance_degradation": {"status": "not_measured"},
        "thresholds": {
            "psi_warning": 0.1,
            "psi_critical": 0.2,
            "categorical_warning": 0.1,
            "categorical_critical": 0.2,
            "min_samples": 30,
        },
        "periods": {"reference": "training", "current": "batch"},
    }

    run_id = log_monitoring_report(
        report,
        tracking_uri=f"sqlite:///{(tmp_path / 'monitoring-mlflow.db').as_posix()}",
    )

    from mlflow.tracking import MlflowClient
    from src.tracking import EXPERIMENT_NAME

    client = MlflowClient(
        tracking_uri=f"sqlite:///{(tmp_path / 'monitoring-mlflow.db').as_posix()}"
    )
    run = client.get_run(run_id)
    assert run.data.tags["monitoring.run_type"] == "drift"
    assert run.data.tags["monitoring.performance_status"] == "not_measured"
    assert run.data.metrics["monitoring.feature_drift.max_statistic"] == pytest.approx(
        0.15
    )
    assert client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id == (
        run.info.experiment_id
    )
    report_path = client.download_artifacts(run_id, "monitoring_report.json")
    assert json.loads(Path(report_path).read_text(encoding="utf-8")) == report


def test_monitoring_api_reports_recent_activity_without_input_features(
    monitoring_database,
):
    database_url, _ = monitoring_database
    app = create_app(
        ARTIFACT_DIR,
        database_url=database_url,
        persist_predictions=True,
    )

    with TestClient(app) as client:
        prediction = client.post(
            "/predict",
            json={
                "model_name": "logistic_regression",
                "features": {
                    "recency_days": 45,
                    "orders_90d": 3,
                    "total_spend": 250.0,
                    "support_tickets": 1,
                },
            },
        )
        summary = client.get("/monitoring/summary")
        activity = client.get("/monitoring/predictions?limit=10")

    assert prediction.status_code == 200
    assert set(prediction.json()) == {
        "prediction",
        "churn_probability",
        "model_name",
        "model_version",
    }
    assert summary.status_code == 200
    assert summary.json()["performance_degradation"]["status"] == "not_measured"
    assert summary.json()["data_drift"]["logistic_regression"]["status"] == (
        "insufficient_data"
    )
    assert (
        summary.json()["prediction_drift"]["logistic_regression"]["reference_rows"]
        == 2000
    )
    assert (
        summary.json()["prediction_drift"]["logistic_regression"]["current_rows"]
        == 1
    )
    assert summary.json()["holdout_metrics"]["logistic_regression"]["source"].endswith(
        "not production"
    )
    assert activity.status_code == 200
    row = activity.json()["predictions"][0]
    assert row["model_name"] == "logistic_regression"
    assert "customer_id" not in row
    assert not set(FEATURES).intersection(row)


def test_monitoring_api_requires_database_when_persistence_is_disabled():
    app = create_app(ARTIFACT_DIR)
    with TestClient(app) as client:
        response = client.get("/monitoring/summary")

    assert response.status_code == 503
