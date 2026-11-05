"""Run repeatable synthetic drift monitoring and log a report to MLflow."""

import argparse
import json
import logging
import os
from pathlib import Path

import mlflow
import pandas as pd

from src.api.service import ModelRegistry
from src.features import FEATURES
from src.ml import make_data
from src.monitoring import MonitoringThresholds, build_monitoring_report
from src.tracking import EXPERIMENT_NAME

LOGGER = logging.getLogger(__name__)
ARTIFACT_DIR = Path("artifacts")


def _load_features(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing = set(FEATURES).difference(data.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required feature columns: "
            + ", ".join(sorted(missing))
        )
    return data


def _shift_feature(data: pd.DataFrame, feature: str | None, amount: float):
    if feature is None:
        return data
    if feature not in FEATURES:
        raise ValueError(f"Unknown feature {feature!r}; expected one of {FEATURES}.")
    shifted = data.copy()
    shifted[feature] = shifted[feature] + amount
    if feature == "recency_days":
        shifted[feature] = shifted[feature].clip(0, 180)
    elif feature in {"orders_90d", "support_tickets"}:
        shifted[feature] = shifted[feature].clip(lower=0).round().astype(int)
    elif feature == "total_spend":
        shifted[feature] = shifted[feature].clip(lower=0)
    return shifted


def log_monitoring_report(report: dict, tracking_uri: str | None = None) -> str:
    previous_tracking_uri = mlflow.get_tracking_uri()
    selected_tracking_uri = (
        tracking_uri
        or os.environ.get("MLFLOW_TRACKING_URI")
        or "sqlite:///mlflow.db"
    )
    mlflow.set_tracking_uri(selected_tracking_uri)
    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name="synthetic-drift-monitoring") as run:
            mlflow.set_tag("monitoring.run_type", "drift")
            mlflow.set_tag(
                "monitoring.performance_status",
                report["performance_degradation"]["status"],
            )
            mlflow.log_params(
                {
                    "monitoring.reference_rows": report["data_drift"][
                        "reference_rows"
                    ],
                    "monitoring.current_rows": report["data_drift"]["current_rows"],
                    "monitoring.reference_period": report["periods"]["reference"],
                    "monitoring.current_period": report["periods"]["current"],
                    **{
                        f"monitoring.threshold.{key}": value
                        for key, value in report["thresholds"].items()
                    },
                }
            )
            feature_values = [
                feature["value"]
                for feature in report["data_drift"]["features"]
                if feature["value"] is not None
            ]
            if feature_values:
                mlflow.log_metric(
                    "monitoring.feature_drift.max_statistic",
                    max(feature_values),
                )
            for model_name, drift in report["prediction_drift"].items():
                value = drift["probability"]["value"]
                if value is not None:
                    mlflow.log_metric(
                        f"monitoring.prediction_drift.{model_name}.probability_psi",
                        value,
                    )
                class_value = drift["class_distribution"]["value"]
                if class_value is not None:
                    mlflow.log_metric(
                        f"monitoring.prediction_drift.{model_name}.class_tvd",
                        class_value,
                    )
            mlflow.log_dict(report, "monitoring_report.json")
            run_id = run.info.run_id
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)
    return run_id


def run_monitoring(
    reference_path: Path,
    current_path: Path | None = None,
    batch_size: int = 500,
    seed: int = 2026,
    shift_feature: str | None = None,
    shift_amount: float = 60.0,
    evaluate_current_labels: bool = False,
    artifact_dir: Path = ARTIFACT_DIR,
    thresholds: MonitoringThresholds | None = None,
):
    reference_data = _load_features(reference_path)
    if current_path is None:
        if evaluate_current_labels:
            raise ValueError(
                "Labeled evaluation requires a user-supplied --current CSV."
            )
        current_data = make_data(n=batch_size, seed=seed)
        current_source = f"synthetic batch (seed={seed})"
    else:
        current_data = pd.read_csv(current_path)
        current_source = current_path.name
    missing = set(FEATURES).difference(current_data.columns)
    if missing:
        raise ValueError(
            "Current data is missing required feature columns: "
            + ", ".join(sorted(missing))
        )
    if evaluate_current_labels and "churned" not in current_data.columns:
        raise ValueError(
            "--evaluate-current-labels requires a binary 'churned' column."
        )
    current_data = _shift_feature(current_data, shift_feature, shift_amount)
    reference_features = reference_data[FEATURES]
    current_features = current_data[FEATURES]

    registry = ModelRegistry(artifact_dir)
    registry.load()
    if not registry.ready:
        raise RuntimeError(f"No trained model artifacts found in {artifact_dir}.")
    reference_predictions = {
        model_name: registry.predict_probabilities(
            model_name, reference_features
        )
        for model_name in registry.models
    }
    current_predictions = {
        model_name: registry.predict_probabilities(model_name, current_features)
        for model_name in registry.models
    }
    comparison_path = artifact_dir / "comparison_metrics.json"
    comparison = (
        json.loads(comparison_path.read_text(encoding="utf-8"))
        if comparison_path.is_file()
        else {}
    )
    report = build_monitoring_report(
        reference_features,
        current_features,
        reference_predictions,
        current_predictions,
        comparison.get("models", {}),
        thresholds,
        (
            current_data["churned"].astype(int).tolist()
            if evaluate_current_labels
            else None
        ),
    )
    report["periods"]["reference"] = reference_path.name
    report["periods"]["current"] = current_source
    report["batch"] = {
        "seed": seed if current_path is None else None,
        "shift_feature": shift_feature,
        "shift_amount": shift_amount if shift_feature else None,
    }
    return report


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=ARTIFACT_DIR / "synthetic_customers.csv",
        help="Stable training reference CSV.",
    )
    parser.add_argument(
        "--current",
        type=Path,
        help="Optional current batch CSV; otherwise generate a synthetic batch.",
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--shift-feature",
        choices=FEATURES,
        help="Optionally apply a numeric shift to the generated batch for a demo.",
    )
    parser.add_argument("--shift-amount", type=float, default=60.0)
    parser.add_argument(
        "--evaluate-current-labels",
        action="store_true",
        help=(
            "Compare a user-supplied labeled current CSV with the synthetic "
            "training holdout; label provenance is not independently verified."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_DIR / "monitoring_report.json",
    )
    parser.add_argument("--tracking-uri", help="MLflow tracking URI.")
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Write the report without creating an MLflow run.",
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    report = run_monitoring(
        args.reference,
        args.current,
        args.batch_size,
        args.seed,
        args.shift_feature,
        args.shift_amount,
        args.evaluate_current_labels,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    run_id = None if args.no_mlflow else log_monitoring_report(
        report, args.tracking_uri
    )
    print(json.dumps(report, indent=2))
    print(f"Saved monitoring report to {args.output}")
    if run_id:
        print(f"MLflow monitoring run ID: {run_id}")
    LOGGER.info(
        "Completed monitoring run: feature_drift=%s",
        report["data_drift"]["status"],
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
