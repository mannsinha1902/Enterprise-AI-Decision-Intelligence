import json
import logging
import pickle
from pathlib import Path

import joblib
import numpy as np

from src.torch_model import load_torch_model

LOGGER = logging.getLogger(__name__)

MODEL_FILES = {
    "logistic_regression": "churn_model.joblib",
    "xgboost": "xgboost_model.joblib",
    "pytorch": "pytorch_model.pt",
}


class ModelRegistry:
    def __init__(self, artifact_dir):
        self.artifact_dir = Path(artifact_dir)
        self.models = {}
        self.model_loading_status = {
            name: "missing" for name in MODEL_FILES
        }
        self.model_version = None

    def load(self):
        metadata_path = self.artifact_dir / "model_metadata.json"
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                version = metadata.get("model_version")
                if isinstance(version, str) and version:
                    self.model_version = version
            except (OSError, ValueError):
                LOGGER.exception("Could not read model version metadata")

        for name, filename in MODEL_FILES.items():
            model_path = self.artifact_dir / filename
            if not model_path.is_file():
                self.model_loading_status[name] = "missing"
                continue
            try:
                if name == "pytorch":
                    model = load_torch_model(model_path)
                else:
                    model = joblib.load(model_path)
                    if not callable(getattr(model, "predict_proba", None)):
                        raise ValueError("model does not provide predict_proba")
            except (
                OSError,
                EOFError,
                ImportError,
                KeyError,
                pickle.UnpicklingError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
                LOGGER.exception("Could not load model artifact %s", model_path)
                self.model_loading_status[name] = "error"
                continue
            self.models[name] = model
            self.model_loading_status[name] = "loaded"

    @property
    def ready(self):
        return bool(self.models)

    def predict_probability(self, model_name, features):
        model = self.models[model_name]
        probability = float(model.predict_proba(features)[0, 1])
        if not 0 <= probability <= 1:
            raise RuntimeError(f"Model {model_name!r} returned an invalid probability")
        return probability

    def predict_probabilities(self, model_name, features):
        model = self.models[model_name]
        probabilities = np.asarray(model.predict_proba(features), dtype=float)
        if (
            probabilities.ndim != 2
            or probabilities.shape[1] < 2
            or not np.isfinite(probabilities[:, 1]).all()
            or ((probabilities[:, 1] < 0) | (probabilities[:, 1] > 1)).any()
        ):
            raise RuntimeError(f"Model {model_name!r} returned invalid probabilities")
        return probabilities[:, 1].astype(float).tolist()
