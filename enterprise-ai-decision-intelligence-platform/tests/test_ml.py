from sklearn.model_selection import train_test_split

from src.ml import compare_models, make_data, train, FEATURES


def test_dataset_reproducible():
    assert make_data(100, 42).equals(make_data(100, 42))


def test_training():
    data = make_data(600)
    model, metrics = train(data)
    assert 0 <= metrics["roc_auc"] <= 1
    assert len(model.predict_proba(data[FEATURES].head())) == 5


def test_model_comparison_uses_one_stratified_holdout():
    data = make_data(600)
    models, comparison = compare_models(data, seed=7)
    _, expected_test = train_test_split(
        data,
        stratify=data["churned"],
        test_size=0.25,
        random_state=7,
    )
    expected_positive_rows = int(expected_test["churned"].sum())

    assert set(models) == {"logistic_regression", "xgboost", "pytorch"}
    assert comparison["holdout"] == {
        "method": "stratified train/test split",
        "random_state": 7,
        "test_size": 0.25,
        "train_rows": 450,
        "test_rows": 150,
        "test_positive_rows": expected_positive_rows,
        "test_negative_rows": len(expected_test) - expected_positive_rows,
    }
    assert (
        comparison["holdout"]["test_positive_rows"]
        + comparison["holdout"]["test_negative_rows"]
        == comparison["holdout"]["test_rows"]
    )

    for name, model in models.items():
        metrics = comparison["models"][name]
        assert 0 <= metrics["roc_auc"] <= 1
        assert 0 <= metrics["average_precision"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1
        assert sum(sum(row) for row in metrics["confusion_matrix"]) == 150
        assert len(model.predict_proba(data[FEATURES].head())) == 5

    assert comparison["training"]["pytorch"]["best_epoch"] <= comparison[
        "training"
    ]["pytorch"]["epochs"]
