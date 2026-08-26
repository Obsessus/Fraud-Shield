"""Candidate model: XGBoost gradient-boosted trees.

Trained on the same leakage-safe temporal features as the baseline (DECISIONS D5) so the
comparison is fair. `scale_pos_weight` is resolved from the training positive rate when set
to ``"auto"`` (no random oversampling, which would leak). Early stopping uses the temporal
validation split.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fraudintel.models.baseline import evaluate, save_model, select_features


def resolve_scale_pos_weight(params: dict[str, Any], y: pd.Series) -> dict[str, Any]:
    """Replace ``scale_pos_weight: "auto"`` with ``n_neg / n_pos`` (XGBoost convention)."""
    params = dict(params)
    if params.get("scale_pos_weight") == "auto":
        pos = int(np.asarray(y).sum())
        neg = int(len(y) - pos)
        params["scale_pos_weight"] = neg / pos if pos else 1.0
    return params


def train_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    seed: int = 42,
    params: dict[str, Any] | None = None,
    early_stopping: int = 50,
) -> Any:
    from xgboost import XGBClassifier

    base = {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.02,
        "subsample": 0.8,
        "colsample_bytree": 0.4,
        "eval_metric": "logloss",
        "n_jobs": -1,
        "random_state": seed,
    }
    if params:
        base.update(params)
    base = resolve_scale_pos_weight(base, y)

    clf = XGBClassifier(**base)
    if X_val is not None and y_val is not None:
        # xgboost 3.x configures early stopping via the constructor and verbose via fit.
        clf.set_params(early_stopping_rounds=early_stopping)
        clf.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
    else:
        clf.fit(X, y)
    return clf


__all__ = ["evaluate", "resolve_scale_pos_weight", "save_model", "select_features", "train_xgboost"]
