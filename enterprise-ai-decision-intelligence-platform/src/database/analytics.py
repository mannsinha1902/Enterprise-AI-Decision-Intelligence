import json

from sqlalchemy import case, func, select

from src.database.models import Customer, Transaction
from src.database.session import create_database_engine, create_session_factory
from src.features import FEATURES


def create_database_analytics(database_url: str):
    def execute(query_function):
        engine = create_database_engine(database_url)
        try:
            factory = create_session_factory(engine)
            with factory() as session:
                return query_function(session)
        finally:
            engine.dispose()

    def customer_summary() -> str:
        churn_value = case((Customer.churned, 1), else_=0)
        columns = [
            func.count(Customer.customer_id),
            func.coalesce(func.sum(churn_value), 0),
            func.avg(churn_value),
            *(func.avg(getattr(Customer, feature)) for feature in FEATURES),
        ]
        row = execute(lambda session: session.execute(select(*columns)).one())
        customer_count, churned_count, churn_rate, *averages = row
        summary = {
            "customer_count": int(customer_count),
            "churned_customers": int(churned_count),
            "churn_rate": float(churn_rate or 0),
            "feature_averages": {
                feature: float(value or 0)
                for feature, value in zip(FEATURES, averages)
            },
        }
        return json.dumps(summary, sort_keys=True)

    def churn_by_recency() -> str:
        churn_value = case((Customer.churned, 1.0), else_=0.0)
        band = case(
            (Customer.recency_days < 30, "0-29"),
            (Customer.recency_days < 60, "30-59"),
            (Customer.recency_days < 90, "60-89"),
            (Customer.recency_days < 120, "90-119"),
            (Customer.recency_days < 150, "120-149"),
            else_="150-180",
        ).label("recency_days")
        query = (
            select(
                band,
                func.count(Customer.customer_id),
                func.avg(churn_value),
            )
            .group_by(band)
            .order_by(band)
        )
        rows = execute(lambda session: session.execute(query).all())
        return json.dumps(
            [
                {
                    "recency_days": row[0],
                    "customer_count": int(row[1]),
                    "churn_rate": float(row[2] or 0),
                }
                for row in rows
            ],
            sort_keys=True,
        )

    def transaction_summary() -> str:
        query = select(
            func.count(Transaction.transaction_id),
            func.coalesce(func.sum(Transaction.amount), 0),
        )
        transaction_count, total_spend = execute(
            lambda session: session.execute(query).one()
        )
        return json.dumps(
            {
                "transaction_count": int(transaction_count),
                "transaction_amount_total": float(total_spend),
            },
            sort_keys=True,
        )

    return customer_summary, churn_by_recency, transaction_summary
