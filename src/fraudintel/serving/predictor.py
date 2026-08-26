"""Serving predictor: loads the champion model and explains predictions (Stage 12)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from fraudintel.explain.shap_explainer import ShapExplainer

DEFAULT_THRESHOLD = 0.5


def decision_for(score: float, threshold: float) -> str:
    """Map a score to an action. ``review`` when at/above the operating threshold."""
    return "review" if score >= threshold else "allow"


class FraudPredictor:
    """Loads the champion model, scores feature vectors, and returns risk factors.

    The model is trained on engineered features, so the serving contract accepts the
    engineered ``features`` vector (not a raw transaction). Real-time aggregate features
    (e.g. ``card1_hist_fraud_rate``) are produced by an upstream feature store; the API's
    responsibility is scoring + explanation (see D21).
    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        threshold: float = DEFAULT_THRESHOLD,
        model_name: str = "xgboost",
        top_k: int = 5,
    ) -> None:
        self.model = model
        self.feature_names = list(feature_names)
        self.threshold = float(threshold)
        self.model_name = model_name
        self.top_k = top_k
        self.explainer = ShapExplainer(model, self.feature_names)

    @classmethod
    def load(
        cls, model_path: Path, threshold_path: Path | None = None, model_name: str = "xgboost"
    ) -> FraudPredictor:
        model = joblib.load(model_path)
        feature_names = list(model.feature_names_in_)
        threshold = cls._load_threshold(threshold_path)
        return cls(model, feature_names, threshold, model_name)

    @staticmethod
    def _load_threshold(threshold_path: Path | None) -> float:
        if threshold_path and threshold_path.exists():
            report = json.loads(threshold_path.read_text())
            if "operating_threshold" in report:
                return float(report["operating_threshold"])
        return DEFAULT_THRESHOLD

    def predict(self, items: list[dict[str, float]]) -> list[dict[str, Any]]:
        """Score a batch of feature dicts; return results with risk factors."""
        if not items:
            return []
        X = pd.DataFrame(items)
        X = X.reindex(columns=self.feature_names, fill_value=0.0)
        scores = np.asarray(self.model.predict_proba(X)[:, 1], dtype="float64")
        factors = self.explainer.explain(X, top_k=self.top_k)
        results: list[dict[str, Any]] = []
        for score, facs in zip(scores, factors, strict=False):
            decision = decision_for(score, self.threshold)
            results.append(
                {
                    "score": float(score),
                    "threshold": self.threshold,
                    "decision": decision,
                    "risk_factors": facs,
                }
            )
        return results
