"""Tests for the XGBoost candidate model: training, scale_pos_weight, early stopping."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudintel.models.candidate import resolve_scale_pos_weight, train_xgboost


def _separable(seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 200
    X = pd.DataFrame(
        {
            "f1": np.concatenate([rng.normal(0, 1, n // 2), rng.normal(5, 1, n // 2)]),
            "f2": np.concatenate([rng.normal(0, 1, n // 2), rng.normal(5, 1, n // 2)]),
        }
    )
    y = pd.Series([0] * (n // 2) + [1] * (n // 2))
    return X, y


def test_train_xgboost_separable():
    X, y = _separable()
    clf = train_xgboost(X, y, seed=0)
    proba = clf.predict_proba(X)[:, 1]
    from sklearn.metrics import average_precision_score

    assert average_precision_score(y, proba) > 0.95


def test_scale_pos_weight_auto_resolves():
    y = pd.Series([0] * 90 + [1] * 10)  # 90 neg, 10 pos -> 9.0
    resolved = resolve_scale_pos_weight({"scale_pos_weight": "auto"}, y)
    assert resolved["scale_pos_weight"] == 9.0


def test_train_xgboost_early_stopping_runs():
    X, y = _separable()
    X_val = X * 1.01
    y_val = y
    clf = train_xgboost(X, y, X_val=X_val, y_val=y_val, seed=1, early_stopping=10)
    assert hasattr(clf, "best_iteration")
    proba = clf.predict_proba(X_val)[:, 1]
    from sklearn.metrics import average_precision_score

    assert 0.0 < average_precision_score(y_val, proba) <= 1.0
