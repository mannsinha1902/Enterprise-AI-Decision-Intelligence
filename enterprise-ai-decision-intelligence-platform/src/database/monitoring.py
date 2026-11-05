"""Read-only queries for persisted model prediction monitoring."""

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import ModelPrediction
from src.features import FEATURES


def load_prediction_batch(
    factory: sessionmaker[Session],
    model_name: str,
    limit: int,
):
    columns = [
        getattr(ModelPrediction, feature).label(feature) for feature in FEATURES
    ]
    columns.extend(
        [
            ModelPrediction.churn_probability,
            ModelPrediction.prediction,
            ModelPrediction.model_version,
            ModelPrediction.created_at,
        ]
    )
    statement = (
        select(*columns)
        .where(ModelPrediction.model_name == model_name)
        .order_by(ModelPrediction.created_at.desc(), ModelPrediction.prediction_id.desc())
        .limit(limit)
    )
    with factory() as session:
        rows = session.execute(statement).mappings().all()
    return [dict(row) for row in rows]


def load_recent_prediction_activity(factory: sessionmaker[Session], limit: int):
    statement = (
        select(
            ModelPrediction.model_name,
            ModelPrediction.model_version,
            ModelPrediction.prediction,
            ModelPrediction.churn_probability,
            ModelPrediction.created_at,
        )
        .order_by(ModelPrediction.created_at.desc(), ModelPrediction.prediction_id.desc())
        .limit(limit)
    )
    with factory() as session:
        rows = session.execute(statement).mappings().all()
    return [
        {
            **dict(row),
            "created_at": row["created_at"].isoformat(),
        }
        for row in rows
    ]
