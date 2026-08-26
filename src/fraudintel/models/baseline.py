"""Baseline model: Logistic Regression with temporal evaluation.

The model is intentionally simple and interpretable, providing a floor that any
candidate model (e.g. XGBoost) must beat under the promotion gate (DECISIONS D5/D7).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Raw identifier / leakage-risky columns dropped before modeling.
IDENTITY_DROP = [
    "TransactionID",
    "TransactionDT",
    "isFraud",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
    "DeviceType",
]


def select_features(df: pd.DataFrame, target: str = "isFraud") -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with identifiers/strings removed and NaNs imputed to 0.

    Keeps the leakage-safe engineered features (e.g. `*_hist_*`, `*_freq`) and the
    original numeric `V*`, `C*`, `D*`, `M*` and amount columns.
    """
    drop = [c for c in IDENTITY_DROP if c in df.columns]
    X = df.drop(columns=drop)
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    X = X[num_cols].fillna(0)
    y = df[target].astype("int64")
    return X, y


def train_baseline(
    X: pd.DataFrame, y: pd.Series, seed: int = 42, params: dict[str, Any] | None = None
) -> Pipeline:
    """Fit a scaled Logistic Regression. `params` overrides defaults."""
    base = {"max_iter": 2000, "class_weight": "balanced", "C": 1.0, "random_state": seed}
    if params:
        base.update(params)
    pipe: Pipeline = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(**base))]
    )
    pipe.fit(X, y)
    return pipe


def pr_auc(y_true: np.ndarray[Any, Any], y_score: np.ndarray[Any, Any]) -> float:
    return float(average_precision_score(y_true, y_score))


def evaluate(estimator: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Compute temporal-aware metrics. PR-AUC is primary (DECISIONS D4)."""
    proba = estimator.predict_proba(X)[:, 1]
    return {
        "pr_auc": pr_auc(np.asarray(y), proba),
        "roc_auc": float(roc_auc_score(y, proba)),
        "n": int(len(y)),
        "positives": int(np.asarray(y).sum()),
    }


def save_model(pipe: Pipeline, path: str | Path) -> None:
    import joblib

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)


def load_model(path: str | Path) -> Pipeline:
    import joblib

    return joblib.load(path)
