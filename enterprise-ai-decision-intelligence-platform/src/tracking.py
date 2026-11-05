from pathlib import Path

import mlflow

EXPERIMENT_NAME = "enterprise-ai-decision-intelligence"


def log_comparison(comparison, artifacts, tracking_uri="sqlite:///mlflow.db"):
    previous_tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name="synthetic-churn-model-comparison") as run:
            holdout = comparison["holdout"]
            mlflow.log_params(
                {
                    "evaluation_method": holdout["method"],
                    "random_state": holdout["random_state"],
                    "test_size": holdout["test_size"],
                    "train_rows": holdout["train_rows"],
                    "test_rows": holdout["test_rows"],
                }
            )
            for model_name, model_parameters in comparison.get(
                "training", {}
            ).items():
                mlflow.log_params(
                    {
                        f"{model_name}_{parameter_name}": parameter_value
                        for parameter_name, parameter_value in model_parameters.items()
                        if isinstance(parameter_value, (str, int, float, bool))
                    }
                )
            metrics = {
                f"{model_name}_{metric_name}": float(model_metrics[metric_name])
                for model_name, model_metrics in comparison["models"].items()
                for metric_name in (
                    "roc_auc",
                    "average_precision",
                    "precision",
                    "recall",
                    "f1",
                )
                if metric_name in model_metrics
            }
            mlflow.log_metrics(metrics)
            mlflow.log_dict(comparison, "comparison_metrics.json")
            mlflow.log_dict(
                {"model_version": run.info.run_id},
                "model_metadata.json",
            )
            for artifact_name, artifact_path in artifacts.items():
                mlflow.log_artifact(
                    str(Path(artifact_path)),
                    artifact_path=artifact_name,
                )
            run_id = run.info.run_id
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)
    return run_id
