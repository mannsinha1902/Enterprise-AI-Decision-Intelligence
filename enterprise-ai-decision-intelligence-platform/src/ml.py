"""Synthetic churn demo. Not a validated production model."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features import FEATURES
from src.torch_model import (
    LEARNING_RATE,
    VALIDATION_SIZE,
    train_mlp,
)

def make_data(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    recency = rng.integers(0, 181, n)
    orders = rng.poisson(4, n)
    spend = np.round(rng.gamma(3, 85, n), 2)
    tickets = rng.poisson(1, n)
    logit = -1.8 + 0.025 * recency - 0.25 * orders - 0.002 * spend + 0.35 * tickets
    probability = 1 / (1 + np.exp(-logit))
    return pd.DataFrame({"customer_id": [f"C{i:05d}" for i in range(n)], "recency_days": recency,
                         "orders_90d": orders, "total_spend": spend, "support_tickets": tickets,
                         "churned": rng.binomial(1, probability)})


def train(df, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(df[FEATURES], df["churned"],
                                                         stratify=df["churned"], test_size=0.25,
                                                         random_state=seed)
    pre = ColumnTransformer([("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                                                      ("scale", StandardScaler())]), FEATURES)])
    model = Pipeline([("preprocess", pre), ("classifier", LogisticRegression(max_iter=1000, random_state=seed))])
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    metrics = _evaluate(y_test, scores)
    return model, metrics


def _evaluate(labels, scores):
    predictions = scores >= 0.5
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "classification_report": classification_report(
            labels, predictions, output_dict=True, zero_division=0
        ),
    }


def compare_models(
    df,
    seed=42,
    test_size=0.25,
    torch_epochs=100,
    torch_batch_size=64,
    torch_patience=10,
):
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES],
        df["churned"],
        stratify=df["churned"],
        test_size=test_size,
        random_state=seed,
    )
    preprocessor = ColumnTransformer(
        [("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                               ("scale", StandardScaler())]), FEATURES)]
    )
    models = {
        "logistic_regression": Pipeline(
            [("preprocess", preprocessor),
             ("classifier", LogisticRegression(max_iter=1000, random_state=seed))]
        ),
        "xgboost": XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
        ),
    }

    model_metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_test)[:, 1]
        model_metrics[name] = _evaluate(y_test, scores)

    torch_model, torch_training = train_mlp(
        X_train,
        y_train,
        seed=seed,
        epochs=torch_epochs,
        batch_size=torch_batch_size,
        patience=torch_patience,
        validation_size=VALIDATION_SIZE,
        learning_rate=LEARNING_RATE,
    )
    models["pytorch"] = torch_model
    model_metrics["pytorch"] = _evaluate(
        y_test, torch_model.predict_proba(X_test)[:, 1]
    )

    comparison = {
        "holdout": {
            "method": "stratified train/test split",
            "random_state": seed,
            "test_size": test_size,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "test_positive_rows": int(y_test.sum()),
            "test_negative_rows": int((y_test == 0).sum()),
        },
        "models": model_metrics,
        "training": {"pytorch": torch_training},
    }
    return models, comparison
