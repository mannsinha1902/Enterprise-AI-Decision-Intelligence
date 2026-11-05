import json
import os
from pathlib import Path

import pandas as pd
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import func, inspect, select
from sqlalchemy.exc import IntegrityError
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.database.models import (
    Customer,
    ModelMetadata,
    ModelPrediction,
    Transaction,
)
from src.database.operations import load_synthetic_customers
from src.database.session import create_database_engine, create_session_factory
from src.ml import make_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
VALID_FEATURES = {
    "recency_days": 45,
    "orders_90d": 3,
    "total_spend": 250.0,
    "support_tickets": 1,
}


@pytest.fixture
def migrated_database(tmp_path, monkeypatch):
    database_url = f"sqlite:///{(tmp_path / 'database.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    command.upgrade(config, "head")
    engine = create_database_engine(database_url)
    yield database_url, engine, create_session_factory(engine)
    engine.dispose()


def test_initial_migration_creates_indexed_tables_and_foreign_keys(
    migrated_database,
):
    _, engine, _ = migrated_database
    inspector = inspect(engine)

    assert set(inspector.get_table_names()) >= {
        "customers",
        "transactions",
        "model_predictions",
        "model_metadata",
        "alembic_version",
    }
    assert {
        index["name"] for index in inspector.get_indexes("model_predictions")
    } >= {"ix_model_predictions_created_at", "ix_model_predictions_model_version"}
    assert inspector.get_foreign_keys("transactions")[0]["referred_table"] == "customers"
    assert inspector.get_foreign_keys("model_predictions")[0]["referred_table"] == "customers"


def test_synthetic_loader_is_idempotent_and_preserves_aggregates(
    migrated_database,
):
    _, _, factory = migrated_database
    generated = make_data(100)
    expected_transaction_count = int(generated["orders_90d"].sum())
    expected_spend = round(float(generated["total_spend"].sum()), 2)

    first_counts = load_synthetic_customers(factory, count=100)
    second_counts = load_synthetic_customers(factory, count=100)

    with factory() as session:
        customer_count = session.scalar(select(func.count()).select_from(Customer))
        transaction_count = session.scalar(
            select(func.count()).select_from(Transaction)
        )
        total_amount = session.scalar(select(func.sum(Transaction.amount)))

    assert first_counts == (100, expected_transaction_count)
    assert second_counts == first_counts
    assert customer_count == 100
    assert transaction_count == expected_transaction_count
    assert float(total_amount) == pytest.approx(expected_spend)


def test_api_keeps_response_contract_and_persists_prediction_and_model_metadata(
    migrated_database,
):
    database_url, _, factory = migrated_database
    app = create_app(
        ARTIFACT_DIR,
        database_url=database_url,
        persist_predictions=True,
    )

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "model_name": "logistic_regression",
                "features": VALID_FEATURES,
            },
        )

    assert response.status_code == 200
    assert set(response.json()) == {
        "prediction",
        "churn_probability",
        "model_name",
        "model_version",
    }
    with factory() as session:
        saved_prediction = session.scalar(select(ModelPrediction))
        saved_metadata = session.scalars(select(ModelMetadata)).all()
    assert saved_prediction is not None
    assert saved_prediction.model_name == "logistic_regression"
    assert saved_prediction.recency_days == VALID_FEATURES["recency_days"]
    assert saved_prediction.churn_probability == pytest.approx(
        response.json()["churn_probability"]
    )
    assert any(row.model_name == "logistic_regression" for row in saved_metadata)


def test_prediction_customer_foreign_key_is_enforced(migrated_database):
    _, _, factory = migrated_database
    with pytest.raises(IntegrityError):
        with factory.begin() as session:
            session.add(
                ModelPrediction(
                    customer_id="missing-customer",
                    model_name="logistic_regression",
                    model_version="test-version",
                    recency_days=10,
                    orders_90d=1,
                    total_spend=10,
                    support_tickets=0,
                    churn_probability=0.25,
                    prediction="no_churn",
                )
            )


@pytest.mark.skipif(
    not os.environ.get("POSTGRES_TEST_URL"),
    reason="PostgreSQL integration check requires POSTGRES_TEST_URL.",
)
def test_postgresql_test_database_is_reachable():
    engine = create_database_engine(os.environ["POSTGRES_TEST_URL"])
    try:
        with engine.connect() as connection:
            assert connection.exec_driver_sql("SELECT 1").scalar_one() == 1
            assert set(inspect(connection).get_table_names()) >= {
                "customers",
                "transactions",
                "model_predictions",
                "model_metadata",
            }
    finally:
        engine.dispose()


def test_database_analytics_exposes_only_fixed_aggregate_tools(
    migrated_database, tmp_path
):
    database_url, _, factory = migrated_database
    csv_path = tmp_path / "synthetic.csv"
    customers = make_data(100)
    customers.to_csv(csv_path, index=False)
    load_synthetic_customers(factory, count=100)
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(
        json.dumps({"holdout": {"test_rows": 1}, "models": {}}),
        encoding="utf-8",
    )

    from src.analytics_agent import create_analytics_tools

    tools = create_analytics_tools(
        csv_path, comparison_path, database_url=database_url
    )
    by_name = {registered_tool.name: registered_tool for registered_tool in tools}
    summary = json.loads(by_name["get_customer_summary"].invoke({}))
    recency = json.loads(by_name["get_churn_by_recency"].invoke({}))
    transactions = json.loads(by_name["get_transaction_summary"].invoke({}))

    assert set(by_name) == {
        "get_customer_summary",
        "get_churn_by_recency",
        "get_model_comparison",
        "get_transaction_summary",
    }
    assert summary["customer_count"] == 100
    assert summary["churn_rate"] == pytest.approx(
        float(customers["churned"].mean())
    )
    assert sum(row["customer_count"] for row in recency) == 100
    assert transactions["transaction_count"] == int(customers["orders_90d"].sum())
    assert transactions["transaction_amount_total"] == pytest.approx(
        float(customers["total_spend"].sum())
    )
