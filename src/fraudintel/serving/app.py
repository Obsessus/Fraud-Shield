"""FastAPI service for fraud scoring (Stage 12).

Endpoints:
  GET  /health            -> service + model status
  POST /predict           -> single transaction scoring + risk factors
  POST /predict/batch     -> batch scoring

The champion model is loaded lazily on first request and cached. Override
``get_predictor`` via ``app.dependency_overrides`` in tests.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from prometheus_fastapi_instrumentator import Instrumentator

from fraudintel.data.paths import artifacts_dir
from fraudintel.serving.features_demo import FeatureHistory
from fraudintel.serving.metrics import PREDICTIONS, record_results
from fraudintel.serving.predictor import FraudPredictor
from fraudintel.serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    DemoFactor,
    DemoFeaturesRequest,
    DemoFeaturesResponse,
    HealthResponse,
    PredictRequest,
    PredictResult,
)
from fraudintel.serving.ui import UI_HTML

MODEL_PATH = artifacts_dir() / "models" / "xgboost_model.joblib"
THRESHOLD_PATH = artifacts_dir() / "models" / "promotion_report.json"
SAMPLE_PATH = artifacts_dir() / "ui" / "sample_features.parquet"
SHAP_PATH = artifacts_dir() / "explainability" / "global_shap_importance.json"
FEATURE_HISTORY_PATH = artifacts_dir() / "ui" / "feature_history.parquet"
TOP_FEATURES_N = 12

_predictor: FraudPredictor | None = None
_sample_df: Any = None
_feature_history = FeatureHistory(FEATURE_HISTORY_PATH)


def _load_top_features() -> list[str]:
    try:
        data = json.loads(Path(SHAP_PATH).read_text(encoding="utf-8"))
        return [f["feature"] for f in data.get("top_features", [])[:TOP_FEATURES_N]]
    except Exception:  # noqa: BLE001 - UI degrades gracefully if SHAP artifact is absent
        return []


TOP_FEATURES = _load_top_features()


def get_predictor() -> FraudPredictor:
    global _predictor
    if _predictor is None:
        try:
            _predictor = FraudPredictor.load(MODEL_PATH, THRESHOLD_PATH)
        except Exception as exc:  # noqa: BLE001 - surface a clean 503 instead of 500
            raise HTTPException(status_code=503, detail=f"model unavailable: {exc}") from None
    return _predictor


app = FastAPI(title="Fraud Intelligence API", version="1.0.0")


@app.get("/", response_class=HTMLResponse)
def ui_root() -> str:
    return UI_HTML


@app.get("/sample")
def sample() -> dict[str, Any]:
    """Return a random real transaction (all engineered features) for the UI demo."""
    global _sample_df
    if _sample_df is None:
        if not SAMPLE_PATH.exists():
            raise HTTPException(status_code=503, detail="UI sample data not available")
        import pandas as pd

        _sample_df = pd.read_parquet(SAMPLE_PATH)
    i = random.randint(0, len(_sample_df) - 1)
    row = _sample_df.iloc[i]
    label = int(row.get("isFraud", 0))
    features: dict[str, float] = {}
    for c in _sample_df.columns:
        if c == "isFraud":
            continue
        try:
            val = float(row[c])
        except (ValueError, TypeError):
            # Non-numeric columns are skipped; the predictor backfills with 0.
            continue
        if not math.isfinite(val):
            val = 0.0
        features[c] = val
    return {
        "transaction_id": f"sample-{i}",
        "label": label,
        "features": features,
        "top_features": TOP_FEATURES,
        "feature_demo": _feature_history.has_data(),
    }


@app.get("/health", response_model=HealthResponse)
def health(predictor: FraudPredictor = Depends(get_predictor)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model=predictor.model_name,
        features=len(predictor.feature_names),
        threshold=predictor.threshold,
    )


@app.post("/predict", response_model=PredictResult)
def predict(
    req: PredictRequest, predictor: FraudPredictor = Depends(get_predictor)
) -> PredictResult:
    result: dict[str, Any] = predictor.predict([req.features])[0]
    PREDICTIONS.labels(endpoint="predict").inc()
    record_results([result])
    return PredictResult(transaction_id=req.transaction_id, **result)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(
    req: BatchPredictRequest, predictor: FraudPredictor = Depends(get_predictor)
) -> BatchPredictResponse:
    results = predictor.predict([item.features for item in req.items])
    PREDICTIONS.labels(endpoint="predict_batch").inc(len(results))
    record_results(results)
    mapped = [
        PredictResult(transaction_id=item.transaction_id, **res)
        for item, res in zip(req.items, results, strict=False)
    ]
    return BatchPredictResponse(results=mapped)


@app.post("/demo/features", response_model=DemoFeaturesResponse)
def demo_features(req: DemoFeaturesRequest) -> DemoFeaturesResponse:
    """Recompute the historical-aggregate features for a raw transaction (UI demo).

    Given the current feature vector plus raw ``card1`` / ``addr1`` / ``amount``
    fields, returns an updated feature vector with the entity-history features
    resolved from the training history, plus a human-readable derivation log.
    """
    if not _feature_history.has_data():
        raise HTTPException(status_code=503, detail="feature history not available")
    updated, derivation = _feature_history.recompute(
        req.features, card1=req.card1, addr1=req.addr1, amount=req.amount
    )
    factors = [
        DemoFactor(feature=d["feature"], value=d["value"], note=d["note"])
        for d in derivation
    ]
    return DemoFeaturesResponse(features=updated, derivation=factors)


Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
