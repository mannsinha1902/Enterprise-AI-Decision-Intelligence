from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, status
from sqlalchemy.exc import SQLAlchemyError

from src.api.schemas import (
    HealthResponse,
    ModelsResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.api.service import MODEL_FILES, ModelRegistry
from src.database.monitoring import (
    load_prediction_batch,
    load_recent_prediction_activity,
)
from src.database.operations import persist_prediction, upsert_model_metadata
from src.database.session import create_database_engine, create_session_factory
from src.features import FEATURES
from src.monitoring import (
    MonitoringThresholds,
    feature_drift_report,
    prediction_drift_report,
)

LOGGER = logging.getLogger(__name__)


def _holdout_metrics_for_monitoring(metrics):
    return {
        model_name: {
            "source": "synthetic stratified training holdout; not production",
            **values,
        }
        for model_name, values in metrics.items()
    }


def _persistence_enabled(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        value = os.environ.get("PERSIST_PREDICTIONS", "false")
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("PERSIST_PREDICTIONS must be a boolean value")


@asynccontextmanager
async def lifespan(app):
    registry = ModelRegistry(Path(app.state.artifact_dir))
    registry.load()
    app.state.model_registry = registry
    app.state.database_engine = None
    app.state.database_session_factory = None
    app.state.monitoring_reference = None
    app.state.monitoring_reference_predictions = {}
    app.state.holdout_metrics = {}
    reference_path = Path(app.state.artifact_dir) / "synthetic_customers.csv"
    comparison_path = Path(app.state.artifact_dir) / "comparison_metrics.json"
    try:
        if reference_path.is_file():
            reference = pd.read_csv(reference_path)
            missing = set(FEATURES).difference(reference.columns)
            if missing:
                raise ValueError(
                    "Monitoring reference is missing feature columns: "
                    + ", ".join(sorted(missing))
                )
            app.state.monitoring_reference = reference[FEATURES]
            app.state.monitoring_reference_predictions = {
                name: registry.predict_probabilities(
                    name, app.state.monitoring_reference
                )
                for name in registry.models
            }
        else:
            LOGGER.warning(
                "Monitoring reference dataset is unavailable: %s", reference_path
            )
        if comparison_path.is_file():
            app.state.holdout_metrics = json.loads(
                comparison_path.read_text(encoding="utf-8")
            ).get("models", {})
    except (OSError, ValueError, KeyError, RuntimeError):
        LOGGER.exception("Could not load monitoring reference data")
    try:
        if app.state.persist_predictions:
            if not app.state.database_url:
                raise RuntimeError(
                    "PERSIST_PREDICTIONS is enabled but DATABASE_URL is not set."
                )
            engine = create_database_engine(app.state.database_url)
            app.state.database_engine = engine
            factory = create_session_factory(engine)
            app.state.database_session_factory = factory
            if registry.model_version:
                for model_name in registry.models:
                    upsert_model_metadata(
                        factory,
                        model_name,
                        registry.model_version,
                        MODEL_FILES[model_name],
                    )
        yield
    finally:
        if app.state.database_engine is not None:
            app.state.database_engine.dispose()


def create_app(
    artifact_dir="artifacts",
    database_url: str | None = None,
    persist_predictions: str | bool | None = None,
):
    persistence_enabled = _persistence_enabled(persist_predictions)
    app = FastAPI(
        title="Enterprise AI Decision Intelligence API",
        description="Local churn predictions from trained synthetic-data models.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.artifact_dir = Path(artifact_dir)
    app.state.database_url = database_url or os.environ.get("DATABASE_URL")
    app.state.persist_predictions = persistence_enabled
    app.state.monitoring_thresholds = MonitoringThresholds.from_environment()

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        registry = request.app.state.model_registry
        return {
            "status": "ready" if registry.ready else "not_ready",
            "model_loading_status": registry.model_loading_status,
            "available_models": list(registry.models),
        }

    @app.get("/models", response_model=ModelsResponse)
    def models(request: Request):
        registry = request.app.state.model_registry
        return {
            "models": [
                {"name": name, "version": registry.model_version}
                for name in registry.models
            ]
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request):
        registry = request.app.state.model_registry
        if not registry.ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No trained models are available. Run train.py first.",
            )
        if payload.model_name not in registry.models:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"Model {payload.model_name!r} is not available. "
                    "Run train.py to train and save all models."
                ),
            )

        features = pd.DataFrame(
            [payload.features.model_dump()],
            columns=FEATURES,
        )
        probability = registry.predict_probability(payload.model_name, features)
        prediction = "churn" if probability >= 0.5 else "no_churn"
        if request.app.state.persist_predictions:
            try:
                persist_prediction(
                    request.app.state.database_session_factory,
                    payload.features.model_dump(),
                    probability,
                    prediction,
                    payload.model_name,
                    registry.model_version,
                )
            except SQLAlchemyError as error:
                LOGGER.exception("Could not persist model prediction")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Prediction persistence is unavailable.",
                ) from error
        return PredictionResponse(
            prediction=prediction,
            churn_probability=probability,
            model_name=payload.model_name,
            model_version=registry.model_version,
        )

    @app.get("/monitoring/summary")
    def monitoring_summary(
        request: Request,
        limit: int = Query(default=1000, ge=1, le=5000),
    ):
        factory = request.app.state.database_session_factory
        if factory is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database-backed monitoring is not enabled.",
            )
        reference = request.app.state.monitoring_reference
        if reference is None:
            return {
                "status": "unavailable",
                "reason": "The stable training reference dataset is unavailable.",
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
                "data_drift": {"status": "unavailable", "features": []},
                "prediction_drift": {},
                "performance_degradation": {
                    "status": "not_measured",
                    "reason": "Current labeled production outcomes are not available.",
                },
                "holdout_metrics": _holdout_metrics_for_monitoring(
                    request.app.state.holdout_metrics
                ),
            }
        try:
            data_drift = {}
            prediction_drift = {}
            model_statuses = []
            for model_name in request.app.state.model_registry.models:
                rows = load_prediction_batch(factory, model_name, limit)
                current_features = pd.DataFrame(
                    [{feature: row[feature] for feature in FEATURES} for row in rows],
                    columns=FEATURES,
                )
                current_probabilities = [
                    float(row["churn_probability"]) for row in rows
                ]
                model_data_drift = feature_drift_report(
                    reference,
                    current_features,
                    request.app.state.monitoring_thresholds,
                )
                model_prediction_drift = prediction_drift_report(
                    request.app.state.monitoring_reference_predictions.get(
                        model_name, []
                    ),
                    current_probabilities,
                    request.app.state.monitoring_thresholds,
                )
                data_drift[model_name] = model_data_drift
                prediction_drift[model_name] = model_prediction_drift
                model_statuses.extend(
                    [model_data_drift["status"], model_prediction_drift["status"]]
                )
            recent = load_recent_prediction_activity(factory, 50)
        except SQLAlchemyError as error:
            LOGGER.exception("Could not query monitoring data")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Monitoring data is unavailable.",
            ) from error

        overall_status = (
            "critical"
            if "critical" in model_statuses
            else "warning"
            if "warning" in model_statuses
            else "ok"
            if model_statuses and all(value == "ok" for value in model_statuses)
            else "insufficient_data"
        )
        return {
            "status": overall_status,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "thresholds": request.app.state.monitoring_thresholds.__dict__,
            "periods": {
                "reference": "training reference: synthetic_customers.csv",
                "current": "most recent persisted predictions per model",
                "reference_rows": int(len(reference)),
                "current_limit_per_model": limit,
            },
            "data_drift": data_drift,
            "prediction_drift": prediction_drift,
            "performance_degradation": {
                "status": "not_measured",
                "reason": (
                    "Persisted predictions do not contain verified outcome labels. "
                    "Drift is not evidence of performance degradation."
                ),
            },
            "holdout_metrics": _holdout_metrics_for_monitoring(
                request.app.state.holdout_metrics
            ),
            "recent_predictions": recent,
        }

    @app.get("/monitoring/predictions")
    def monitoring_predictions(
        request: Request,
        limit: int = Query(default=50, ge=1, le=500),
    ):
        factory = request.app.state.database_session_factory
        if factory is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database-backed monitoring is not enabled.",
            )
        try:
            return {
                "predictions": load_recent_prediction_activity(factory, limit)
            }
        except SQLAlchemyError as error:
            LOGGER.exception("Could not query recent prediction activity")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Monitoring data is unavailable.",
            ) from error

    return app


app = create_app()
