from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ModelName = Literal["logistic_regression", "xgboost", "pytorch"]


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    recency_days: int = Field(ge=0, le=180)
    orders_90d: int = Field(ge=0)
    total_spend: float = Field(ge=0)
    support_tickets: int = Field(ge=0)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: CustomerFeatures
    model_name: ModelName = "logistic_regression"


class PredictionResponse(BaseModel):
    prediction: Literal["churn", "no_churn"]
    churn_probability: float = Field(ge=0, le=1)
    model_name: ModelName
    model_version: str | None


class ModelInfo(BaseModel):
    name: ModelName
    version: str | None


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class HealthResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    model_loading_status: dict[str, Literal["loaded", "missing", "error"]]
    available_models: list[ModelName]
