import time

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sklearn.datasets import make_classification
from xgboost import XGBClassifier

from fraudintel.data.paths import artifacts_dir
from fraudintel.serving.app import app, get_predictor
from fraudintel.serving.predictor import FraudPredictor, decision_for
from fraudintel.serving.schemas import PredictRequest


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


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["features"] == 2
    assert body["threshold"] == 0.5


def test_predict_returns_score_and_factors(client):
    r = client.post("/predict", json={"transaction_id": "t1", "features": {"f0": 1.0, "f1": 0.0}})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["score"] <= 1.0
    assert body["decision"] in ("review", "allow")
    assert body["transaction_id"] == "t1"
    assert len(body["risk_factors"]) >= 1
    for rf in body["risk_factors"]:
        assert rf["feature"] in ("f0", "f1")
        assert isinstance(rf["contribution"], float)
        assert isinstance(rf["value"], float)


def test_predict_handles_extra_and_missing_features(client):
    r = client.post("/predict", json={"features": {"f0": 0.5, "unknown_feature": 9.0}})
    assert r.status_code == 200
    assert 0.0 <= r.json()["score"] <= 1.0


def test_predict_batch(client):
    payload = {"items": [{"features": {"f0": 1.0}}, {"features": {"f1": -1.0}}]}
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2


def test_predictor_loads_champion_if_present():
    model_path = artifacts_dir() / "models" / "xgboost_model.joblib"
    report_path = artifacts_dir() / "models" / "promotion_report.json"
    if not model_path.exists():
        pytest.skip("champion artifact not present")
    predictor = FraudPredictor.load(model_path, report_path)
    assert len(predictor.feature_names) > 0
    sample = {predictor.feature_names[0]: 0.0}
    res = predictor.predict([sample])[0]
    assert 0.0 <= res["score"] <= 1.0
    assert res["decision"] in ("review", "allow")


def test_schema_rejects_non_finite_value():
    # Non-finite values are rejected by the schema validator (the JSON layer also
    # rejects NaN/Infinity tokens at the wire, a lower-level defense).
    with pytest.raises(ValidationError):
        PredictRequest(features={"f0": float("nan")})
    with pytest.raises(ValidationError):
        PredictRequest(features={"f0": float("inf")})


def test_predict_rejects_wrong_type_over_http(client):
    # A non-numeric feature value is a type error -> 422 at the API boundary.
    r = client.post("/predict", json={"features": {"f0": "not-a-number"}})
    assert r.status_code == 422


def test_predict_with_empty_features_returns_base_score(client):
    r = client.post("/predict", json={"features": {}})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["score"] <= 1.0
    assert body["decision"] in ("review", "allow")


def test_predict_batch_empty_returns_empty(client):
    r = client.post("/predict/batch", json={"items": []})
    assert r.status_code == 200
    assert r.json()["results"] == []


def test_risk_factors_within_top_k(client):
    r = client.post("/predict", json={"features": {"f0": 0.5}}).json()
    assert len(r["risk_factors"]) <= 5


def test_batch_performance_sanity(client):
    payload = {"items": [{"features": {"f0": 0.5, "f1": -0.5}} for _ in range(200)]}
    t0 = time.time()
    r = client.post("/predict/batch", json=payload)
    elapsed = time.time() - t0
    assert r.status_code == 200
    assert elapsed < 2.0


def test_decision_for_boundary():
    assert decision_for(0.5, 0.5) == "review"
    assert decision_for(0.499, 0.5) == "allow"
    assert decision_for(0.798, 0.798) == "review"


UI_SAMPLE_PRESENT = (artifacts_dir() / "ui" / "sample_features.parquet").exists()


def test_ui_root_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert "Fraud Intelligence" in r.text


@pytest.mark.skipif(not UI_SAMPLE_PRESENT, reason="UI sample data not present")
def test_sample_endpoint_returns_real_transaction(client):
    r = client.get("/sample")
    assert r.status_code == 200
    body = r.json()
    assert body["transaction_id"]
    assert body["label"] in (0, 1)
    assert isinstance(body["features"], dict) and len(body["features"]) > 0
    assert isinstance(body["top_features"], list)


@pytest.mark.skipif(not UI_SAMPLE_PRESENT, reason="UI sample data not present")
def test_sample_includes_feature_demo_flag(client):
    body = client.get("/sample").json()
    assert "feature_demo" in body


FEATURE_HISTORY_PRESENT = (artifacts_dir() / "ui" / "feature_history.parquet").exists()


@pytest.mark.skipif(not FEATURE_HISTORY_PRESENT, reason="feature history not present")
def test_demo_features_recomputes(client):
    sample = client.get("/sample").json()
    payload = {"features": sample["features"], "card1": 1000.0, "addr1": 300.0, "amount": 123.45}
    r = client.post("/demo/features", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["features"], dict)
    assert isinstance(body["derivation"], list) and len(body["derivation"]) > 0
    assert body["features"]["TransactionAmt"] == 123.45
    assert "TransactionAmt_log" in body["features"]
