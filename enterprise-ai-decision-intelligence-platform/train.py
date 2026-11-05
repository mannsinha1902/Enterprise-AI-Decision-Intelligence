import json
from pathlib import Path
import joblib
from src.ml import compare_models, make_data
from src.tracking import log_comparison
from src.torch_model import save_torch_model

if __name__ == "__main__":
    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    df = make_data()
    df.to_csv(out / "synthetic_customers.csv", index=False)
    models, comparison = compare_models(df)
    model = models["logistic_regression"]
    metrics = comparison["models"]["logistic_regression"]
    joblib.dump(model, out / "churn_model.joblib")
    joblib.dump(models["xgboost"], out / "xgboost_model.joblib")
    save_torch_model(
        models["pytorch"],
        out / "pytorch_model.pt",
        comparison["training"]["pytorch"],
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "comparison_metrics.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    run_id = log_comparison(
        comparison,
        {
            "baseline_model": out / "churn_model.joblib",
            "xgboost_model": out / "xgboost_model.joblib",
            "pytorch_model": out / "pytorch_model.pt",
            "synthetic_data": out / "synthetic_customers.csv",
        },
    )
    (out / "model_metadata.json").write_text(
        json.dumps({"model_version": run_id}, indent=2),
        encoding="utf-8",
    )
    print("Saved model and metrics to artifacts/")
    print(f"Stratified holdout: {comparison['holdout']['test_rows']} test rows")
    for name, model_metrics in comparison["models"].items():
        print(
            f"{name} ROC-AUC: {model_metrics['roc_auc']:.3f}; "
            f"average precision: {model_metrics['average_precision']:.3f}; "
            f"precision: {model_metrics['precision']:.3f}; "
            f"recall: {model_metrics['recall']:.3f}; "
            f"F1: {model_metrics['f1']:.3f}"
        )
    print(f"MLflow run ID: {run_id} (local SQLite tracking store: mlflow.db)")
