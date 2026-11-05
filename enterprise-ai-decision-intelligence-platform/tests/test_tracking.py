import json

from mlflow.tracking import MlflowClient

from src.ml import compare_models, make_data
from src.tracking import EXPERIMENT_NAME, log_comparison


def test_comparison_is_logged_to_local_mlflow(tmp_path):
    models, comparison = compare_models(
        make_data(300), torch_epochs=12, torch_patience=4
    )
    artifact_paths = {}
    for model_name in models:
        artifact = tmp_path / f"{model_name}.joblib"
        artifact.write_text("synthetic model artifact", encoding="utf-8")
        artifact_paths[model_name] = artifact
    tracking_db = (tmp_path / "tracking.db").as_posix()
    tracking_uri = f"sqlite:///{tracking_db}"

    run_id = log_comparison(comparison, artifact_paths, tracking_uri=tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    assert client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id == run.info.experiment_id
    assert run.data.params["evaluation_method"] == "stratified train/test split"
    assert run.data.metrics["logistic_regression_roc_auc"] == comparison["models"][
        "logistic_regression"
    ]["roc_auc"]
    assert run.data.metrics["xgboost_average_precision"] == comparison["models"][
        "xgboost"
    ]["average_precision"]
    assert run.data.metrics["pytorch_f1"] == comparison["models"]["pytorch"]["f1"]
    assert run.data.params["pytorch_batch_size"] == "64"
    assert run.data.params["pytorch_hidden_sizes"] == "32,16"

    logged_json = client.download_artifacts(run_id, "comparison_metrics.json")
    with open(logged_json, encoding="utf-8") as file:
        assert json.load(file) == comparison
    logged_metadata = client.download_artifacts(run_id, "model_metadata.json")
    with open(logged_metadata, encoding="utf-8") as file:
        assert json.load(file) == {"model_version": run_id}
    for model_name in models:
        logged_models = client.list_artifacts(run_id, model_name)
        assert [artifact.path for artifact in logged_models] == [
            f"{model_name}/{model_name}.joblib"
        ]
