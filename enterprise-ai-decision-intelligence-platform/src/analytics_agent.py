"""Read-only, local LangChain tools for synthetic customer analytics."""

import json
import os
from pathlib import Path

import pandas as pd
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool

from src.database.analytics import create_database_analytics
from src.features import FEATURES

DEFAULT_DATA_PATH = Path("artifacts/synthetic_customers.csv")
DEFAULT_COMPARISON_PATH = Path("artifacts/comparison_metrics.json")
SYSTEM_PROMPT = """You are a read-only business analytics assistant.
Use only the supplied analytics tools to answer questions about the synthetic
customer data and evaluated models. Do not invent results, access other data,
write files, execute code or SQL, or claim synthetic results describe real
customers. If the tools cannot answer a question, say so."""

_CUSTOMER_COLUMNS = set(FEATURES) | {"churned"}
_MODEL_METRICS = (
    "roc_auc",
    "average_precision",
    "precision",
    "recall",
    "f1",
)


def _read_customer_data(data_path: Path) -> pd.DataFrame:
    customers = pd.read_csv(data_path)
    missing_columns = _CUSTOMER_COLUMNS.difference(customers.columns)
    if missing_columns:
        raise ValueError(
            "Synthetic customer data is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    return customers


def create_analytics_tools(
    data_path: str | Path = DEFAULT_DATA_PATH,
    comparison_path: str | Path = DEFAULT_COMPARISON_PATH,
    database_url: str | None = None,
):
    """Create tools scoped to the project's synthetic data and model metrics."""
    data_file = Path(data_path)
    comparison_file = Path(comparison_path)
    configured_database_url = (
        database_url
        if database_url is not None
        else os.environ.get("DATABASE_URL")
    )
    database_analytics = (
        create_database_analytics(configured_database_url)
        if configured_database_url
        else None
    )

    @tool
    def get_customer_summary() -> str:
        """Return aggregate counts, churn rate, and feature averages."""
        if database_analytics is not None:
            return database_analytics[0]()
        customers = _read_customer_data(data_file)
        summary = {
            "customer_count": int(len(customers)),
            "churned_customers": int(customers["churned"].sum()),
            "churn_rate": float(customers["churned"].mean()),
            "feature_averages": {
                feature: float(customers[feature].mean()) for feature in FEATURES
            },
        }
        return json.dumps(summary, sort_keys=True)

    @tool
    def get_churn_by_recency() -> str:
        """Return customer counts and churn rates for fixed recency-day bands."""
        if database_analytics is not None:
            return database_analytics[1]()
        customers = _read_customer_data(data_file)
        bands = pd.cut(
            customers["recency_days"],
            bins=[0, 30, 60, 90, 120, 150, 181],
            labels=["0-29", "30-59", "60-89", "90-119", "120-149", "150-180"],
            right=False,
        )
        grouped = customers.assign(recency_band=bands).groupby(
            "recency_band", observed=False
        )
        results = [
            {
                "recency_days": str(name),
                "customer_count": int(len(group)),
                "churn_rate": float(group["churned"].mean()),
            }
            for name, group in grouped
            if len(group)
        ]
        return json.dumps(results, sort_keys=True)

    @tool
    def get_model_comparison() -> str:
        """Return holdout evaluation metrics for the trained synthetic-data models."""
        comparison = json.loads(comparison_file.read_text(encoding="utf-8"))
        models = comparison.get("models")
        if not isinstance(models, dict):
            raise ValueError("Model comparison data does not contain a models object")
        summary = {
            "holdout": {
                key: comparison["holdout"][key]
                for key in ("method", "test_rows")
                if key in comparison.get("holdout", {})
            },
            "models": {
                name: {
                    metric: float(metrics[metric])
                    for metric in _MODEL_METRICS
                    if metric in metrics
                }
                for name, metrics in models.items()
                if isinstance(metrics, dict)
            },
        }
        return json.dumps(summary, sort_keys=True)

    tools = [get_customer_summary, get_churn_by_recency, get_model_comparison]
    if database_analytics is not None:

        @tool
        def get_transaction_summary() -> str:
            """Return aggregate counts and total amount from synthetic transactions."""
            return database_analytics[2]()

        tools.append(get_transaction_summary)
    return tools


def create_analytics_agent(
    model: BaseChatModel | None = None,
    data_path: str | Path = DEFAULT_DATA_PATH,
    comparison_path: str | Path = DEFAULT_COMPARISON_PATH,
    database_url: str | None = None,
):
    """Build the agent; the default provider is a local Ollama server."""
    if model is None:
        from langchain_ollama import ChatOllama

        model = ChatOllama(
            model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
            base_url=os.environ.get(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            ),
            temperature=0,
        )
    return create_agent(
        model=model,
        tools=create_analytics_tools(
            data_path, comparison_path, database_url=database_url
        ),
        system_prompt=SYSTEM_PROMPT,
    )


def ask_analytics_question(
    question: str,
    model: BaseChatModel | None = None,
    data_path: str | Path = DEFAULT_DATA_PATH,
    comparison_path: str | Path = DEFAULT_COMPARISON_PATH,
    database_url: str | None = None,
) -> str:
    """Answer one business question using only the read-only local tools."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string")
    if len(question) > 1000:
        raise ValueError("Question must be 1000 characters or fewer")

    result = create_analytics_agent(
        model=model,
        data_path=data_path,
        comparison_path=comparison_path,
        database_url=database_url,
    ).invoke({"messages": [{"role": "user", "content": question.strip()}]})
    answer = result["messages"][-1].content
    if isinstance(answer, str):
        return answer
    if isinstance(answer, list) and all(
        isinstance(block, dict) and isinstance(block.get("text"), str)
        for block in answer
    ):
        return "".join(block["text"] for block in answer)
    raise TypeError("Analytics agent returned an unsupported answer format")
