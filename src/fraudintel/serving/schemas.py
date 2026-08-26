"""Pydantic request/response schemas for the Fraud Intelligence API (Stage 12)."""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, field_validator


class RiskFactor(BaseModel):
    """A feature contribution driving a prediction (SHAP value)."""

    feature: str
    contribution: float = Field(description="SHAP contribution to the fraud score")
    value: float = Field(description="Feature value observed for this transaction")


class PredictRequest(BaseModel):
    """Scoring request. ``features`` is the engineered feature vector the model was
    trained on (see D21 — raw transactions are transformed upstream by a feature store).
    """

    transaction_id: str | None = Field(default=None, description="Optional caller id")
    features: dict[str, float] = Field(description="Engineered model features")

    @field_validator("features")
    @classmethod
    def _finite_features(cls, v: dict[str, float]) -> dict[str, float]:
        for name, val in v.items():
            if not math.isfinite(val):
                raise ValueError(f"feature '{name}' must be finite, got {val}")
        return v


class PredictResult(BaseModel):
    transaction_id: str | None = None
    score: float = Field(description="Fraud probability in [0, 1]")
    threshold: float = Field(description="Operating threshold used for the decision")
    decision: str = Field(description="'review' if score >= threshold else 'allow'")
    risk_factors: list[RiskFactor]


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest]


class BatchPredictResponse(BaseModel):
    results: list[PredictResult]


class HealthResponse(BaseModel):
    status: str
    model: str
    features: int
    threshold: float


class DemoFactor(BaseModel):
    """A derived history feature and how it was computed (UI demo log)."""

    feature: str
    value: float
    note: str


class DemoFeaturesRequest(BaseModel):
    """Real-time feature recomputation request (UI demo).

    Sends the current engineered feature vector plus the raw identity/amount fields
    the user edited; the service resolves the entity-history features from the
    training history and returns the updated vector.
    """

    features: dict[str, float] = Field(description="Current engineered feature vector")
    card1: float | None = None
    addr1: float | None = None
    amount: float | None = None


class DemoFeaturesResponse(BaseModel):
    features: dict[str, float] = Field(description="Updated engineered feature vector")
    derivation: list[DemoFactor]
