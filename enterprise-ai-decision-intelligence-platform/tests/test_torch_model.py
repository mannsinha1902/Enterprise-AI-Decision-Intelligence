import numpy as np

from src.features import FEATURES
from src.ml import make_data
from src.torch_model import load_torch_model, save_torch_model, train_mlp


def test_torch_training_predictions_and_serialization_are_reproducible(tmp_path):
    data = make_data(240)
    features = data[FEATURES]
    labels = data["churned"]
    first_model, training_info = train_mlp(
        features, labels, seed=13, epochs=16, batch_size=32, patience=4
    )
    second_model, _ = train_mlp(
        features, labels, seed=13, epochs=16, batch_size=32, patience=4
    )

    first_predictions = first_model.predict_proba(features.head(10))
    second_predictions = second_model.predict_proba(features.head(10))

    assert first_predictions.shape == (10, 2)
    assert np.all((first_predictions >= 0) & (first_predictions <= 1))
    np.testing.assert_allclose(first_predictions, second_predictions, rtol=0, atol=0)
    assert 1 <= training_info["best_epoch"] <= training_info["epochs_run"] <= 16
    assert training_info["training_rows"] + training_info["validation_rows"] == 240

    artifact = tmp_path / "pytorch_model.pt"
    save_torch_model(first_model, artifact, training_info)
    restored_model = load_torch_model(artifact)
    np.testing.assert_allclose(
        first_predictions,
        restored_model.predict_proba(features.head(10)),
        rtol=0,
        atol=0,
    )
