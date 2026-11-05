"""Statistical drift monitoring for synthetic model reference and current batches."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from scipy.stats import chi2_contingency, ks_2samp


@dataclass(frozen=True)
class MonitoringThresholds:
    psi_warning: float = 0.1
    psi_critical: float = 0.2
    categorical_warning: float = 0.1
    categorical_critical: float = 0.2
    min_samples: int = 30
    performance_warning_drop: float = 0.1
    performance_critical_drop: float = 0.2

    def __post_init__(self):
        for warning, critical in (
            (self.psi_warning, self.psi_critical),
            (self.categorical_warning, self.categorical_critical),
            (self.performance_warning_drop, self.performance_critical_drop),
        ):
            if (
                not math.isfinite(warning)
                or not math.isfinite(critical)
                or not 0 <= warning < critical
            ):
                raise ValueError("Drift thresholds must satisfy 0 <= warning < critical.")
        if self.min_samples < 2:
            raise ValueError("Monitoring minimum sample count must be at least 2.")

    @classmethod
    def from_environment(cls):
        return cls(
            psi_warning=float(os.getenv("MONITORING_PSI_WARNING_THRESHOLD", "0.1")),
            psi_critical=float(os.getenv("MONITORING_PSI_CRITICAL_THRESHOLD", "0.2")),
            categorical_warning=float(
                os.getenv("MONITORING_CATEGORICAL_WARNING_THRESHOLD", "0.1")
            ),
            categorical_critical=float(
                os.getenv("MONITORING_CATEGORICAL_CRITICAL_THRESHOLD", "0.2")
            ),
            min_samples=int(os.getenv("MONITORING_MIN_SAMPLES", "30")),
            performance_warning_drop=float(
                os.getenv("MONITORING_PERFORMANCE_WARNING_DROP_THRESHOLD", "0.1")
            ),
            performance_critical_drop=float(
                os.getenv("MONITORING_PERFORMANCE_CRITICAL_DROP_THRESHOLD", "0.2")
            ),
        )


def _status(value: float, warning: float, critical: float, sample_sizes, minimum):
    if any(size < minimum for size in sample_sizes):
        return "insufficient_data"
    if value >= critical:
        return "critical"
    if value >= warning:
        return "warning"
    return "ok"


def _overall_status(statuses: list[str]) -> str:
    if not statuses:
        return "unavailable"
    for candidate in ("critical", "warning"):
        if candidate in statuses:
            return candidate
    if all(status == "ok" for status in statuses):
        return "ok"
    if all(status == "insufficient_data" for status in statuses):
        return "insufficient_data"
    if all(status == "unavailable" for status in statuses):
        return "unavailable"
    return "insufficient_data"


def _numeric_drift(
    reference: pd.Series,
    current: pd.Series,
    thresholds: MonitoringThresholds,
) -> dict[str, Any]:
    reference_values = pd.to_numeric(reference, errors="coerce").to_numpy(dtype=float)
    current_values = pd.to_numeric(current, errors="coerce").to_numpy(dtype=float)
    reference_values = reference_values[np.isfinite(reference_values)]
    current_values = current_values[np.isfinite(current_values)]
    reference_missing_rate = 1 - len(reference_values) / max(len(reference), 1)
    current_missing_rate = 1 - len(current_values) / max(len(current), 1)
    missing_rate_delta = abs(current_missing_rate - reference_missing_rate)
    if not len(reference_values) or not len(current_values):
        return {
            "type": "numeric",
            "statistic": "psi",
            "value": None,
            "ks_p_value": None,
            "reference_missing_rate": float(reference_missing_rate),
            "current_missing_rate": float(current_missing_rate),
            "missing_rate_delta": float(missing_rate_delta),
            "status": "insufficient_data",
        }

    quantiles = np.unique(
        np.quantile(reference_values, np.linspace(0, 1, min(10, len(reference_values)) + 1))
    )
    edges = np.concatenate(([-np.inf], quantiles[1:-1], [np.inf]))
    reference_counts = np.histogram(reference_values, bins=edges)[0].astype(float)
    current_counts = np.histogram(current_values, bins=edges)[0].astype(float)
    epsilon = 1e-6
    reference_proportions = np.clip(
        reference_counts / reference_counts.sum(), epsilon, None
    )
    current_proportions = np.clip(
        current_counts / current_counts.sum(), epsilon, None
    )
    reference_proportions /= reference_proportions.sum()
    current_proportions /= current_proportions.sum()
    psi = float(
        np.sum(
            (current_proportions - reference_proportions)
            * np.log(current_proportions / reference_proportions)
        )
    )
    status_value = max(psi, missing_rate_delta)
    return {
        "type": "numeric",
        "statistic": "psi",
        "value": psi,
        "ks_p_value": float(ks_2samp(reference_values, current_values).pvalue),
        "reference_missing_rate": float(reference_missing_rate),
        "current_missing_rate": float(current_missing_rate),
        "missing_rate_delta": float(missing_rate_delta),
        "status": _status(
            status_value,
            thresholds.psi_warning,
            thresholds.psi_critical,
            (len(reference_values), len(current_values)),
            thresholds.min_samples,
        ),
    }


def _categorical_drift(
    reference: pd.Series,
    current: pd.Series,
    thresholds: MonitoringThresholds,
) -> dict[str, Any]:
    reference_values = reference.astype("string").fillna("<missing>")
    current_values = current.astype("string").fillna("<missing>")
    if not len(reference_values) or not len(current_values):
        return {
            "type": "categorical",
            "statistic": "total_variation_distance",
            "value": None,
            "chi_square_p_value": None,
            "reference_categories": int(reference_values.nunique()),
            "current_categories": int(current_values.nunique()),
            "status": "insufficient_data",
        }
    categories = sorted(set(reference_values) | set(current_values))
    reference_counts = reference_values.value_counts().reindex(categories, fill_value=0)
    current_counts = current_values.value_counts().reindex(categories, fill_value=0)
    reference_proportions = reference_counts.to_numpy(dtype=float) / max(
        len(reference_values), 1
    )
    current_proportions = current_counts.to_numpy(dtype=float) / max(
        len(current_values), 1
    )
    total_variation = float(
        0.5 * np.abs(current_proportions - reference_proportions).sum()
    )
    if len(categories) > 1:
        p_value = float(
            chi2_contingency(
                np.vstack((reference_counts.to_numpy(), current_counts.to_numpy())),
                correction=False,
            ).pvalue
        )
    else:
        p_value = 1.0
    return {
        "type": "categorical",
        "statistic": "total_variation_distance",
        "value": total_variation,
        "chi_square_p_value": p_value,
        "reference_categories": int(reference_values.nunique()),
        "current_categories": int(current_values.nunique()),
        "status": _status(
            total_variation,
            thresholds.categorical_warning,
            thresholds.categorical_critical,
            (len(reference_values), len(current_values)),
            thresholds.min_samples,
        ),
    }


def feature_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    thresholds: MonitoringThresholds | None = None,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or MonitoringThresholds()
    features = feature_columns or list(reference.columns)
    reports = []
    for feature in features:
        if feature not in reference.columns or feature not in current.columns:
            reports.append(
                {
                    "feature": feature,
                    "type": "unknown",
                    "statistic": None,
                    "value": None,
                    "status": "unavailable",
                }
            )
            continue
        if pd.api.types.is_numeric_dtype(reference[feature]):
            report = _numeric_drift(reference[feature], current[feature], thresholds)
        else:
            report = _categorical_drift(reference[feature], current[feature], thresholds)
        report["feature"] = feature
        reports.append(report)
    return {
        "status": _overall_status([report["status"] for report in reports]),
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "features": reports,
    }


def prediction_drift_report(
    reference_probabilities,
    current_probabilities,
    thresholds: MonitoringThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or MonitoringThresholds()
    reference = pd.Series(reference_probabilities, dtype=float)
    current = pd.Series(current_probabilities, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    probability = _numeric_drift(reference, current, thresholds)
    reference_classes = pd.Series(
        np.where(reference >= 0.5, "churn", "no_churn"), dtype="string"
    )
    current_classes = pd.Series(
        np.where(current >= 0.5, "churn", "no_churn"), dtype="string"
    )
    classes = _categorical_drift(reference_classes, current_classes, thresholds)
    return {
        "status": _overall_status([probability["status"], classes["status"]]),
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "probability": probability,
        "class_distribution": classes,
        "reference_churn_rate": float((reference >= 0.5).mean()) if len(reference) else None,
        "current_churn_rate": float((current >= 0.5).mean()) if len(current) else None,
    }


def performance_degradation_report(
    labels,
    probabilities,
    holdout_metrics: dict[str, float],
    thresholds: MonitoringThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or MonitoringThresholds()
    y_true = np.asarray(labels, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if len(y_true) != len(scores):
        raise ValueError("Current labels and prediction probabilities must be aligned.")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Current evaluation labels must be binary values 0 or 1.")
    if len(y_true) < thresholds.min_samples or len(np.unique(y_true)) < 2:
        return {
            "status": "insufficient_data",
            "reason": "At least the minimum sample count and both labels are required.",
            "current_metrics": None,
            "metric_drops": None,
        }

    current_metrics = {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "f1": float(f1_score(y_true, scores >= 0.5, zero_division=0)),
    }
    metric_drops = {
        metric: float(holdout_metrics[metric] - value)
        for metric, value in current_metrics.items()
        if metric in holdout_metrics
    }
    if not metric_drops:
        return {
            "status": "unavailable",
            "reason": "Comparable labeled holdout metrics are unavailable.",
            "current_metrics": current_metrics,
            "metric_drops": {},
        }
    largest_drop = max(metric_drops.values())
    if largest_drop >= thresholds.performance_critical_drop:
        status = "critical"
    elif largest_drop >= thresholds.performance_warning_drop:
        status = "warning"
    else:
        status = "ok"
    return {
        "status": status,
        "basis": (
            "User-supplied labeled current batch compared with the synthetic "
            "training holdout; label provenance is not independently verified."
        ),
        "current_metrics": current_metrics,
        "metric_drops": metric_drops,
        "warning_drop_threshold": thresholds.performance_warning_drop,
        "critical_drop_threshold": thresholds.performance_critical_drop,
    }


def build_monitoring_report(
    reference_features: pd.DataFrame,
    current_features: pd.DataFrame,
    reference_predictions: dict[str, list[float]],
    current_predictions: dict[str, list[float]],
    holdout_metrics: dict[str, dict[str, float]],
    thresholds: MonitoringThresholds | None = None,
    current_labels=None,
) -> dict[str, Any]:
    thresholds = thresholds or MonitoringThresholds()
    data_drift = feature_drift_report(
        reference_features,
        current_features,
        thresholds,
    )
    prediction_drift = {
        model_name: prediction_drift_report(
            reference_predictions.get(model_name, []),
            current_predictions.get(model_name, []),
            thresholds,
        )
        for model_name in sorted(
            set(reference_predictions) | set(current_predictions) | set(holdout_metrics)
        )
    }
    if current_labels is None:
        performance = {
            "status": "not_measured",
            "reason": (
                "Current labeled production outcomes are not available. "
                "Drift is not evidence of performance degradation."
            ),
        }
    else:
        model_performance = {
            model_name: performance_degradation_report(
                current_labels,
                current_predictions.get(model_name, []),
                model_metrics,
                thresholds,
            )
            for model_name, model_metrics in holdout_metrics.items()
        }
        statuses = [result["status"] for result in model_performance.values()]
        performance = {
            "status": (
                "critical"
                if "critical" in statuses
                else "warning"
                if "warning" in statuses
                else "ok"
                if statuses and all(value == "ok" for value in statuses)
                else "insufficient_data"
            ),
            "basis": (
                "User-supplied labeled current batch; label provenance is not "
                "independently verified and results are not a production claim."
            ),
            "models": model_performance,
        }
    return {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": asdict(thresholds),
        "periods": {
            "reference": "training reference dataset",
            "current": "current prediction batch",
        },
        "data_drift": data_drift,
        "prediction_drift": prediction_drift,
        "performance_degradation": performance,
        "holdout_metrics": {
            model_name: {
                "source": "synthetic stratified training holdout; not production",
                **metrics,
            }
            for model_name, metrics in holdout_metrics.items()
        },
    }
