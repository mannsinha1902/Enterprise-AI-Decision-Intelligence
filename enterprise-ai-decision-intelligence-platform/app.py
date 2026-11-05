import json
import logging
import numpy as np
import joblib
import os
import pandas as pd
import streamlit as st
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from src.explain import explain_xgboost
from src.analytics_agent import ask_analytics_question
from src.ml import FEATURES
from src.torch_model import load_torch_model

LOGGER = logging.getLogger(__name__)
st.set_page_config(page_title="Enterprise AI Decision Intelligence", layout="wide")


@st.cache_data(ttl=30)
def _load_monitoring_summary(api_url):
    endpoint = f"{api_url.rstrip('/')}/monitoring/summary"
    try:
        with urlopen(endpoint, timeout=5) as response:
            return json.loads(response.read())
    except HTTPError as error:
        raise RuntimeError(
            f"Monitoring API returned HTTP {error.code}."
        ) from error
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError("Monitoring data could not be loaded from the API.") from error

st.title("Enterprise AI Decision Intelligence")
st.caption("Portfolio demonstration using synthetic customer data — not production predictions")
st.subheader("Local prediction API")
st.caption("Start the local FastAPI service separately to use the trained models via REST.")
st.link_button("Open API documentation", "http://127.0.0.1:8000/docs")
st.code(
    """POST http://127.0.0.1:8000/predict
{
  "model_name": "pytorch",
  "features": {
    "recency_days": 45,
    "orders_90d": 3,
    "total_spend": 250.0,
    "support_tickets": 1
  }
}""",
    language="json",
)
with st.expander("Read-only business analytics assistant"):
    st.caption(
        "Uses a local Ollama model and read-only tools over synthetic customer "
        "aggregates and held-out model metrics. Start Ollama locally before asking."
    )
    with st.form("analytics_question_form"):
        analytics_question = st.text_input(
            "Ask about the synthetic customers or model comparison",
            max_chars=1000,
        )
        ask_question = st.form_submit_button("Ask analytics agent")
    if ask_question:
        if not analytics_question.strip():
            st.error("Enter a business analytics question.")
        else:
            with st.spinner("Querying local synthetic-data analytics..."):
                analytics_answer = ask_analytics_question(analytics_question)
            st.text(analytics_answer)
path = Path("artifacts")
if not (path / "churn_model.joblib").exists():
    st.warning("First run: python train.py")
    st.stop()
model = joblib.load(path / "churn_model.joblib")
data = pd.read_csv(path / "synthetic_customers.csv")
data["churn_probability"] = model.predict_proba(data[FEATURES])[:, 1]
comparison_path = path / "comparison_metrics.json"
if comparison_path.exists():
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
left, middle, right = st.columns(3)
left.metric("Customers", f"{len(data):,}")
middle.metric("Mean predicted churn risk", f"{data['churn_probability'].mean():.1%}")
right.metric("High-risk customers (>= 50%)", int((data["churn_probability"] >= .5).sum()))
st.subheader("Churn risk distribution")
risk_distribution = (
    data.assign(
        risk_band=pd.cut(
            data["churn_probability"],
            bins=[0, .2, .4, .6, .8, 1],
            labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
            include_lowest=True,
        )
    )
    .groupby("risk_band", observed=False)
    .size()
    .rename("Customers")
    .reset_index()
)
st.bar_chart(risk_distribution, x="risk_band", y="Customers")
if comparison_path.exists():
    comparison_rows = [
        {
            "model": (
                "PyTorch"
                if model_name == "pytorch"
                else model_name.replace("_", " ").title()
            ),
            "ROC-AUC": model_metrics["roc_auc"],
            "Average precision": model_metrics["average_precision"],
        }
        for model_name, model_metrics in comparison["models"].items()
    ]
    st.subheader("Model comparison (shared held-out customers)")
    st.bar_chart(
        pd.DataFrame(comparison_rows),
        x="model",
        y=["ROC-AUC", "Average precision"],
    )
    if all(
        "confusion_matrix" in metrics
        for metrics in comparison["models"].values()
    ):
        confusion_rows = []
        for model_name, model_metrics in comparison["models"].items():
            matrix = model_metrics["confusion_matrix"]
            confusion_rows.append(
                {
                    "model": (
                        "PyTorch"
                        if model_name == "pytorch"
                        else model_name.replace("_", " ").title()
                    ),
                    "true negatives": matrix[0][0],
                    "false positives": matrix[0][1],
                    "false negatives": matrix[1][0],
                    "true positives": matrix[1][1],
                }
            )
        st.dataframe(pd.DataFrame(confusion_rows), hide_index=True)
    else:
        st.info("Run train.py to add held-out confusion matrices.")
else:
    st.info("Run train.py to generate held-out model comparison metrics.")

st.subheader("Model monitoring")
monitoring_api_url = os.environ.get("MONITORING_API_URL", "http://127.0.0.1:8000")
try:
    monitoring = _load_monitoring_summary(monitoring_api_url)
except RuntimeError as error:
    LOGGER.exception("Could not load model monitoring summary")
    st.warning(f"Monitoring data is currently unavailable. {error}")
else:
    periods = monitoring.get("periods", {})
    st.caption(
        "Data drift compares the stable training reference with recent persisted "
        "prediction inputs. Prediction drift compares the model's training-reference "
        "scores with recent persisted scores. Drift is not performance degradation."
    )
    st.caption(
        f"Reference: {periods.get('reference', 'unavailable')} "
        f"({periods.get('reference_rows', 'n/a')} rows) | "
        f"Current: {periods.get('current', 'unavailable')}"
    )
    status = monitoring.get("status", "unavailable")
    st.metric("Monitoring alert status", status.replace("_", " ").title())

    drift_by_model = monitoring.get("data_drift", {})
    if drift_by_model:
        monitored_models = list(drift_by_model)
        selected_model = st.selectbox(
            "Model used for current feature comparison",
            monitored_models,
            key="monitoring_model",
        )
        feature_rows = drift_by_model[selected_model].get("features", [])
        if feature_rows:
            feature_table = pd.DataFrame(
                [
                    {
                        "Feature": item["feature"],
                        "Type": item["type"],
                        "Statistic": item["statistic"],
                        "Drift statistic": item["value"],
                        "KS p-value": item.get("ks_p_value"),
                        "Alert": item["status"],
                    }
                    for item in feature_rows
                ]
            )
            st.markdown("**Feature data drift**")
            st.dataframe(feature_table, hide_index=True, use_container_width=True)
            chart_rows = feature_table.dropna(subset=["Drift statistic"])
            if not chart_rows.empty:
                st.bar_chart(
                    chart_rows,
                    x="Feature",
                    y="Drift statistic",
                )
        else:
            st.info("No feature drift metrics are available for this period.")
    else:
        st.info("No model prediction batches are available for drift monitoring yet.")

    prediction_rows = []
    for model_name, drift in monitoring.get("prediction_drift", {}).items():
        prediction_rows.append(
            {
                "Model": model_name,
                "Reference rows": drift["reference_rows"],
                "Current rows": drift["current_rows"],
                "Probability PSI": drift["probability"]["value"],
                "Class total-variation distance": drift["class_distribution"]["value"],
                "Reference predicted churn rate": drift["reference_churn_rate"],
                "Current predicted churn rate": drift["current_churn_rate"],
                "Alert": drift["status"],
            }
        )
    st.markdown("**Prediction distribution drift**")
    if prediction_rows:
        st.dataframe(pd.DataFrame(prediction_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No prediction distribution data is available yet.")

    st.markdown("**Model metrics (synthetic training holdout, not production)**")
    holdout_rows = [
        {
            "Model": model_name,
            "ROC-AUC": metrics.get("roc_auc"),
            "Average precision": metrics.get("average_precision"),
            "Precision": metrics.get("precision"),
            "Recall": metrics.get("recall"),
            "F1": metrics.get("f1"),
        }
        for model_name, metrics in monitoring.get("holdout_metrics", {}).items()
    ]
    if holdout_rows:
        st.dataframe(pd.DataFrame(holdout_rows), hide_index=True, use_container_width=True)
    else:
        st.info("Saved holdout metrics are unavailable.")
    performance = monitoring.get("performance_degradation", {})
    st.info(
        "Performance degradation: "
        + performance.get("status", "not_measured")
        + ". Verified labeled outcomes from a current production period are required "
        "to assess production performance."
    )
    if performance.get("models"):
        performance_rows = [
            {
                "Model": model_name,
                "Status": result["status"],
                "Current ROC-AUC": (result.get("current_metrics") or {}).get(
                    "roc_auc"
                ),
                "Current F1": (result.get("current_metrics") or {}).get("f1"),
            }
            for model_name, result in performance["models"].items()
        ]
        st.dataframe(
            pd.DataFrame(performance_rows),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(performance.get("basis", "User-supplied labeled batch."))

    st.markdown("**Recent prediction activity**")
    recent_predictions = monitoring.get("recent_predictions", [])
    if recent_predictions:
        recent_table = pd.DataFrame(recent_predictions).rename(
            columns={
                "model_name": "Model",
                "model_version": "Model version",
                "prediction": "Prediction",
                "churn_probability": "Churn probability",
                "created_at": "Created at",
            }
        )
        st.dataframe(recent_table, hide_index=True, use_container_width=True)
    else:
        st.info("No persisted predictions are available for the selected period.")

torch_model_path = path / "pytorch_model.pt"
if torch_model_path.exists():
    data["pytorch_probability"] = load_torch_model(
        torch_model_path
    ).predict_proba(data[FEATURES])[:, 1]
st.subheader("XGBoost SHAP explanations")
xgboost_path = path / "xgboost_model.joblib"
if xgboost_path.exists():
    xgboost_model = joblib.load(xgboost_path)
    xgboost_explanation = explain_xgboost(xgboost_model, data[FEATURES])
    data["xgboost_probability"] = xgboost_model.predict_proba(data[FEATURES])[:, 1]

    global_importance = (
        pd.DataFrame(
            {
                "feature": FEATURES,
                "mean_absolute_shap": np.abs(xgboost_explanation.values).mean(axis=0),
            }
        )
        .sort_values("mean_absolute_shap", ascending=False)
        .reset_index(drop=True)
    )
    st.caption(
        "Mean absolute feature impact across customers. SHAP contributions are in "
        "raw model-score (log-odds) units; they describe model behavior, not causation."
    )
    st.bar_chart(global_importance, x="feature", y="mean_absolute_shap")

    example_index = int(data["xgboost_probability"].idxmax())
    st.markdown(
        f"Example customer `{data.loc[example_index, 'customer_id']}` "
        f"(XGBoost churn risk: {data.loc[example_index, 'xgboost_probability']:.1%})"
    )
    local_explanation = pd.DataFrame(
        {
            "feature": FEATURES,
            "shap_contribution": xgboost_explanation.values[example_index],
        }
    ).sort_values("shap_contribution")
    st.bar_chart(local_explanation, x="feature", y="shap_contribution")
else:
    st.info("Run train.py to generate the XGBoost model for SHAP explanations.")
st.subheader("Customer predictions")
st.dataframe(data.sort_values("churn_probability", ascending=False), use_container_width=True)
st.caption("Model scores are illustrative and trained on synthetic labels.")
