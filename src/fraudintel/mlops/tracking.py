"""MLflow experiment tracking for the Fraud Intelligence Platform.

Every model run is logged here so results are reproducible and comparable (DESIGN:
"log every experiment to MLflow"). Both the baseline and candidate training scripts call
`log_model_run`, which records hyperparameters, temporal-split metrics, the fitted model,
and the exact feature list used.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import mlflow
import pandas as pd

EXPERIMENT_NAME = "fraud-intel"


def _log_param(key: str, value: Any) -> None:
    if isinstance(value, (str, int, float, bool)) or value is None:
        mlflow.log_param(key, value)
    else:
        mlflow.log_param(key, str(value))


def _log_feature_artifact(feature_columns: list[str]) -> None:
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w") as fh:
        fh.write("\n".join(feature_columns))
    try:
        mlflow.log_artifact(path, artifact_path="features")
    finally:
        os.remove(path)


def log_model_run(
    model_name: str,
    model: Any,
    params: dict[str, Any],
    metrics: dict[str, Any],
    feature_columns: list[str] | None = None,
    signature_sample: pd.DataFrame | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Log one model run to MLflow and return the run id.

    ``metrics`` may be nested (e.g. ``{"holdout": {"pr_auc": 0.5}}``); nested keys are
    flattened to ``<split>_<metric>``.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=model_name) as run:
        for key, value in (params or {}).items():
            _log_param(key, value)
        for metric_key, metric_value in (metrics or {}).items():
            if isinstance(metric_value, dict):
                for inner_key, inner_value in metric_value.items():
                    mlflow.log_metric(f"{metric_key}_{inner_key}", float(inner_value))
            else:
                mlflow.log_metric(metric_key, float(metric_value))
        if tags:
            mlflow.set_tags(tags)

        log_kwargs: dict[str, Any] = {}
        if signature_sample is not None:
            log_kwargs["input_example"] = signature_sample
        _log_model(model, log_kwargs)

        if feature_columns is not None:
            _log_feature_artifact(feature_columns)
        return str(run.info.run_id)


def _log_model(model: Any, log_kwargs: dict[str, Any]) -> None:
    """Log with the XGBoost flavor for XGBoost estimators, else the sklearn flavor.

    mlflow 3.x' sklearn flavor refuses untrusted (skops) types such as XGBClassifier,
    so XGBoost must use its dedicated flavor.
    """
    model_cls = type(model).__name__
    if "XGB" in model_cls:
        import mlflow.xgboost

        mlflow.xgboost.log_model(model, "model", **log_kwargs)
    else:
        import mlflow.sklearn

        mlflow.sklearn.log_model(model, "model", **log_kwargs)
