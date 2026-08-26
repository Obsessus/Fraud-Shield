"""Prometheus metrics for the Fraud Intelligence API (Stage 17).

Custom business metrics complement the HTTP metrics the FastAPI instrumentator
exposes at ``/metrics`` (request count, latency, etc.).
"""

from __future__ import annotations

from typing import Any

from prometheus_client import Counter, Histogram

PREDICTIONS = Counter(
    "fraud_predictions_total",
    "Total prediction requests served.",
    ["endpoint"],
)
DECISIONS = Counter(
    "fraud_decisions_total",
    "Count of review/allow decisions produced.",
    ["decision"],
)
SCORES = Histogram(
    "fraud_score",
    "Distribution of fraud scores (0..1).",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


def record_results(results: list[dict[str, Any]]) -> None:
    """Record business metrics for a batch of prediction results."""
    for r in results:
        DECISIONS.labels(decision=r["decision"]).inc()
        SCORES.observe(float(r["score"]))
