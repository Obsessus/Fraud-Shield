"""Tests for the MLflow experiment-tracking helper."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression

from fraudintel.mlops.tracking import EXPERIMENT_NAME, log_model_run


def test_log_model_run_records_params_metrics_and_model(tmp_path):
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    y = pd.Series([0, 1, 0])
    model = LogisticRegression().fit(X, y)

    run_id = log_model_run(
        model_name="lr_test",
        model=model,
        params={"seed": 0, "C": 1.0, "scheme": "balanced"},
        metrics={"holdout": {"pr_auc": 0.5, "roc_auc": 0.9}, "validation": {"pr_auc": 0.4}},
        feature_columns=["a", "b"],
        signature_sample=X,
        tags={"model_type": "logistic_regression"},
    )

    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.info.run_id == run_id
    assert run.data.params["seed"] == "0"
    assert run.data.params["scheme"] == "balanced"  # non-scalar coerced to str, not crashed
    assert abs(run.data.metrics["holdout_pr_auc"] - 0.5) < 1e-9
    assert abs(run.data.metrics["validation_pr_auc"] - 0.4) < 1e-9
    # model artifact logged (mlflow logs it under the "model/" prefix)
    model_arts = [a.path for a in client.list_artifacts(run_id, path="model")]
    assert any(p.endswith("MLmodel") for p in model_arts)
