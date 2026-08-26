import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import make_classification
from xgboost import XGBClassifier

from fraudintel.mlops.monitoring import run_drift
from fraudintel.serving.app import app, get_predictor
from fraudintel.serving.predictor import FraudPredictor


@pytest.fixture
def client():
    X, y = make_classification(
        n_samples=300, n_features=2, n_informative=1, n_redundant=0,
        n_repeated=0, n_classes=2, n_clusters_per_class=1, random_state=0,
    )
    model = XGBClassifier(n_estimators=10, max_depth=3, early_stopping_rounds=5, verbose=False)
    model.fit(X, y, eval_set=[(X, y)])
    predictor = FraudPredictor(model, ["f0", "f1"], threshold=0.5)
    app.dependency_overrides[get_predictor] = lambda: predictor
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_metrics_endpoint_exposes_custom_and_http_metrics(client):
    client.post("/predict", json={"features": {"f0": 0.5, "f1": -0.3}})
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "fraud_predictions_total" in body
    assert "fraud_decisions_total" in body
    assert "http_requests_total" in body  # from the FastAPI instrumentator


def test_run_drift_returns_summary_structure():
    ref = pd.DataFrame(
        {
            "a": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
    )
    cur = pd.DataFrame(
        {
            "a": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1],
            "b": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
    )
    summary = run_drift(ref, cur)
    assert "dataset_drift" in summary
    assert summary["total_columns"] == 2
    assert isinstance(summary["details"], list)
    assert len(summary["details"]) == 2
