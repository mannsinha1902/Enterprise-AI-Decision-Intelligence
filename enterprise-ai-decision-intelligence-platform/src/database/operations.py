from datetime import datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy import delete
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import (
    Customer,
    ModelMetadata,
    ModelPrediction,
    Transaction,
    utc_now,
)
from src.features import FEATURES
from src.ml import make_data


def upsert_model_metadata(
    factory: sessionmaker[Session],
    model_name: str,
    model_version: str,
    artifact_file: str,
) -> None:
    now = utc_now()
    with factory.begin() as session:
        existing = session.get(
            ModelMetadata, (model_name, model_version)
        )
        if existing is None:
            session.add(
                ModelMetadata(
                    model_name=model_name,
                    model_version=model_version,
                    model_metadata={"artifact_file": artifact_file},
                    created_at=now,
                    loaded_at=now,
                )
            )
        else:
            existing.model_metadata = {"artifact_file": artifact_file}
            existing.loaded_at = now


def persist_prediction(
    factory: sessionmaker[Session],
    features: dict,
    probability: float,
    prediction: str,
    model_name: str,
    model_version: str | None,
) -> None:
    with factory.begin() as session:
        session.add(
            ModelPrediction(
                model_name=model_name,
                model_version=model_version,
                recency_days=features["recency_days"],
                orders_90d=features["orders_90d"],
                total_spend=Decimal(str(features["total_spend"])).quantize(
                    Decimal("0.01")
                ),
                support_tickets=features["support_tickets"],
                churn_probability=probability,
                prediction=prediction,
            )
        )


def load_synthetic_customers(
    factory: sessionmaker[Session],
    count: int = 2000,
    seed: int = 42,
) -> tuple[int, int]:
    """Generate and idempotently load synthetic customers and derived transactions."""
    data = make_data(n=count, seed=seed)
    required = set(FEATURES) | {"customer_id", "churned"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            "Synthetic CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    customer_count = 0
    transaction_count = 0
    epoch = datetime(2025, 1, 1, tzinfo=timezone.utc)
    with factory.begin() as session:
        for row in data.to_dict(orient="records"):
            customer_id = str(row["customer_id"])
            recency_days = int(row["recency_days"])
            orders_90d = int(row["orders_90d"])
            total_spend = Decimal(str(row["total_spend"])).quantize(
                Decimal("0.01")
            )
            customer = session.get(Customer, customer_id)
            if customer is None:
                customer = Customer(customer_id=customer_id)
                session.add(customer)
            customer.recency_days = recency_days
            customer.orders_90d = orders_90d
            customer.total_spend = total_spend
            customer.support_tickets = int(row["support_tickets"])
            customer.churned = bool(row["churned"])
            customer_count += 1

            session.execute(
                delete(Transaction).where(Transaction.customer_id == customer_id)
            )
            if orders_90d <= 0:
                continue
            cents = int((total_spend * 100).to_integral_value())
            base_cents, remainder = divmod(cents, orders_90d)
            for order_number in range(orders_90d):
                amount_cents = base_cents + (1 if order_number < remainder else 0)
                session.add(
                    Transaction(
                        transaction_id=f"{customer_id}-T{order_number + 1:04d}",
                        customer_id=customer_id,
                        amount=Decimal(amount_cents).scaleb(-2),
                        occurred_at=epoch
                        - timedelta(
                            days=recency_days + order_number * 7
                        ),
                    )
                )
                transaction_count += 1
    return customer_count, transaction_count
