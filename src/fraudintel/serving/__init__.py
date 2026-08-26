"""Serving package: FastAPI fraud-scoring service and predictor."""

from fraudintel.serving.app import app
from fraudintel.serving.predictor import FraudPredictor
from fraudintel.serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResult,
    RiskFactor,
)

__all__ = [
    "app",
    "FraudPredictor",
    "RiskFactor",
    "PredictRequest",
    "PredictResult",
    "BatchPredictRequest",
    "BatchPredictResponse",
    "HealthResponse",
]
