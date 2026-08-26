"""Tests for the baseline model: feature selection, training, and evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudintel.models.baseline import (
    evaluate,
    load_model,
    pr_auc,
    save_model,
    select_features,
    train_baseline,
)


def test_select_features_drops_identifiers_and_strings():
    df = pd.DataFrame(
        {
            "TransactionID": [1],
            "TransactionDT": [1],
            "isFraud": [0],
            "card1": ["A"],
            "addr1": [1],
            "P_emaildomain": ["a"],
            "DeviceInfo": ["x"],
            "DeviceType": ["d"],
            "V1": [1.0],
            "TransactionAmt": [10.0],
            "card1_hist_fraud_rate": [0.2],
            "os_family": ["iOS"],
        }
    )
    X, y = select_features(df)
    assert "isFraud" not in X.columns
    assert "card1" not in X.columns
    assert "os_family" not in X.columns  # string column dropped
    assert "card1_hist_fraud_rate" in X.columns  # engineered numeric kept
    assert "V1" in X.columns
    assert y.iloc[0] == 0
    assert X.isna().sum().sum() == 0  # NaNs imputed


def test_train_baseline_separable():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {
            "f1": np.concatenate([rng.normal(0, 1, n // 2), rng.normal(5, 1, n // 2)]),
            "f2": np.concatenate([rng.normal(0, 1, n // 2), rng.normal(5, 1, n // 2)]),
        }
    )
    y = pd.Series([0] * (n // 2) + [1] * (n // 2))
    pipe = train_baseline(X, y, seed=0)
    m = evaluate(pipe, X, y)
    assert m["pr_auc"] > 0.95
    assert 0.0 <= m["roc_auc"] <= 1.0


def test_pr_auc_in_unit_interval():
    y = np.array([0, 0, 1, 0, 0])
    # positive (idx2, score 0.7) is not the top-scoring row -> AP strictly < 1
    score = np.array([0.2, 0.8, 0.7, 0.3, 0.4])
    assert 0.0 < pr_auc(y, score) < 1.0


def test_model_save_load_roundtrip(tmp_path):
    X = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0], "f2": [1.0, 0.0, 3.0, 2.0]})
    y = pd.Series([0, 0, 1, 1])
    pipe = train_baseline(X, y, seed=1)
    path = tmp_path / "m.joblib"
    save_model(pipe, path)
    loaded = load_model(path)
    assert evaluate(loaded, X, y)["pr_auc"] == evaluate(pipe, X, y)["pr_auc"]
