import json

import pandas as pd
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.analytics_agent import (
    ask_analytics_question,
    create_analytics_agent,
    create_analytics_tools,
)


class MockToolCallingModel(BaseChatModel):
    responses: list[AIMessage]
    bound_tool_names: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "mock-tool-calling"

    def bind_tools(self, tools, **kwargs):
        self.bound_tool_names = [tool.name for tool in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[ChatGeneration(message=self.responses.pop(0))]
        )


@pytest.fixture
def analytics_files(tmp_path):
    customers = pd.DataFrame(
        {
            "customer_id": ["synthetic-1", "synthetic-2", "synthetic-3"],
            "recency_days": [10, 45, 170],
            "orders_90d": [3, 1, 0],
            "total_spend": [120.0, 40.0, 0.0],
            "support_tickets": [0, 2, 1],
            "churned": [0, 1, 1],
        }
    )
    data_path = tmp_path / "synthetic_customers.csv"
    customers.to_csv(data_path, index=False)
    comparison_path = tmp_path / "comparison_metrics.json"
    comparison_path.write_text(
        json.dumps(
            {
                "holdout": {"method": "stratified train/test split", "test_rows": 1},
                "models": {
                    "logistic_regression": {
                        "roc_auc": 0.8,
                        "average_precision": 0.7,
                        "precision": 0.6,
                        "recall": 0.5,
                        "f1": 0.55,
                        "confusion_matrix": [[1, 2], [3, 4]],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return data_path, comparison_path


def test_read_only_tools_return_only_aggregate_data(analytics_files):
    data_path, comparison_path = analytics_files
    summary, recency, comparison = create_analytics_tools(
        data_path, comparison_path
    )

    assert json.loads(summary.invoke({}))["churn_rate"] == pytest.approx(2 / 3)
    recency_result = json.loads(recency.invoke({}))
    assert [row["recency_days"] for row in recency_result] == ["0-29", "30-59", "150-180"]
    comparison_result = json.loads(comparison.invoke({}))
    assert comparison_result["models"]["logistic_regression"]["roc_auc"] == 0.8
    assert "confusion_matrix" not in comparison_result["models"]["logistic_regression"]
    assert "customer_id" not in json.dumps(
        [json.loads(summary.invoke({})), recency_result]
    )


def test_mocked_agent_calls_analytics_tool_without_external_provider(analytics_files):
    data_path, comparison_path = analytics_files
    model = MockToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_customer_summary",
                        "args": {},
                        "id": "summary-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The synthetic churn rate is 66.7%."),
        ]
    )
    agent = create_analytics_agent(
        model=model, data_path=data_path, comparison_path=comparison_path
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is the churn rate?"}]}
    )

    assert model.bound_tool_names == [
        "get_customer_summary",
        "get_churn_by_recency",
        "get_model_comparison",
    ]
    assert result["messages"][-1].content == "The synthetic churn rate is 66.7%."
    assert any(message.name == "get_customer_summary" for message in result["messages"])


@pytest.mark.parametrize("question", ["", "  ", "x" * 1001])
def test_question_validation_rejects_empty_or_oversized_input(question):
    with pytest.raises(ValueError):
        ask_analytics_question(question, model=MockToolCallingModel(responses=[]))


def test_tools_report_missing_synthetic_data(analytics_files, tmp_path):
    _, comparison_path = analytics_files
    summary, _, _ = create_analytics_tools(
        tmp_path / "missing.csv", comparison_path
    )

    with pytest.raises(FileNotFoundError):
        summary.invoke({})
