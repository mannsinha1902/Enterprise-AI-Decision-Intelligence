import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.features import FEATURES

HIDDEN_SIZES = (32, 16)
LEARNING_RATE = 0.001
VALIDATION_SIZE = 0.2


class ChurnMLP(nn.Module):
    def __init__(self, input_size=len(FEATURES)):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZES[0]),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1]),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZES[1], 1),
        )

    def forward(self, features):
        return self.layers(features).squeeze(1)


class TorchChurnPredictor:
    def __init__(self, model, imputer_statistics, scaler_mean, scaler_scale):
        self.model = model.eval()
        self.imputer_statistics = np.asarray(imputer_statistics, dtype=np.float64)
        self.scaler_mean = np.asarray(scaler_mean, dtype=np.float64)
        self.scaler_scale = np.asarray(scaler_scale, dtype=np.float64)

    def predict_proba(self, data):
        values = data[FEATURES].to_numpy(dtype=np.float64)
        values = np.where(np.isnan(values), self.imputer_statistics, values)
        values = (values - self.scaler_mean) / self.scaler_scale
        features = torch.as_tensor(values, dtype=torch.float32, device="cpu")
        with torch.inference_mode():
            positive_probability = torch.sigmoid(self.model(features)).numpy()
        return np.column_stack((1.0 - positive_probability, positive_probability))


def train_mlp(
    features,
    labels,
    seed=42,
    epochs=100,
    batch_size=64,
    patience=10,
    validation_size=VALIDATION_SIZE,
    learning_rate=LEARNING_RATE,
):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if patience < 1:
        raise ValueError("patience must be positive")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    X_fit, X_validation, y_fit, y_validation = train_test_split(
        features[FEATURES],
        labels,
        test_size=validation_size,
        stratify=labels,
        random_state=seed,
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(imputer.fit_transform(X_fit)).astype(
        np.float32
    )
    X_validation_scaled = scaler.transform(
        imputer.transform(X_validation)
    ).astype(np.float32)

    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    training_data = TensorDataset(
        torch.from_numpy(X_fit_scaled),
        torch.as_tensor(np.asarray(y_fit), dtype=torch.float32),
    )
    training_loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_features = torch.from_numpy(X_validation_scaled)
    validation_labels = torch.as_tensor(
        np.asarray(y_validation), dtype=torch.float32
    )

    model = ChurnMLP(input_size=len(FEATURES)).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()
    best_state = None
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    epochs_run = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_features, batch_labels in training_loader:
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = loss_function(logits, batch_labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            validation_loss = loss_function(
                model(validation_features), validation_labels
            ).item()
        epochs_run = epoch
        if validation_loss < best_validation_loss - 1e-6:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("MLP training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    predictor = TorchChurnPredictor(
        model,
        imputer.statistics_,
        scaler.mean_,
        scaler.scale_,
    )
    training_info = {
        "seed": seed,
        "epochs": epochs,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "best_validation_loss": float(best_validation_loss),
        "batch_size": batch_size,
        "patience": patience,
        "validation_size": validation_size,
        "learning_rate": learning_rate,
        "input_size": len(FEATURES),
        "hidden_sizes": ",".join(str(size) for size in HIDDEN_SIZES),
        "training_rows": len(X_fit),
        "validation_rows": len(X_validation),
    }
    return predictor, training_info


def save_torch_model(predictor, path, training_info):
    torch.save(
        {
            "features": FEATURES,
            "hidden_sizes": list(HIDDEN_SIZES),
            "state_dict": predictor.model.state_dict(),
            "imputer_statistics": torch.as_tensor(predictor.imputer_statistics),
            "scaler_mean": torch.as_tensor(predictor.scaler_mean),
            "scaler_scale": torch.as_tensor(predictor.scaler_scale),
            "training_info": training_info,
        },
        Path(path),
    )


def load_torch_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["features"] != FEATURES:
        raise ValueError("PyTorch model features do not match the project schema")
    if tuple(checkpoint["hidden_sizes"]) != HIDDEN_SIZES:
        raise ValueError("PyTorch model architecture does not match this project")

    model = ChurnMLP(input_size=len(FEATURES)).cpu()
    model.load_state_dict(checkpoint["state_dict"])
    return TorchChurnPredictor(
        model,
        checkpoint["imputer_statistics"],
        checkpoint["scaler_mean"],
        checkpoint["scaler_scale"],
    )
