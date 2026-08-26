"""Real-time feature derivation for the UI demo.

Given a transaction's raw identity fields (``card1``, ``addr1``, ``TransactionAmt``),
recompute the historical-aggregate features the model expects. This mirrors the
offline logic in ``features/build.py`` (``add_test_entity_features``): the history
features for an entity key are the *global train aggregates* for that key. A key
with no training history is treated as a brand-new entity (all zeros), which is
exactly how the offline ``merge(..., how="left")`` + ``fillna(0)`` behaves.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from fraudintel.data.paths import artifacts_dir

HISTORY_KEYS = ("card1", "addr1")
HISTORY_PATH = artifacts_dir() / "ui" / "feature_history.parquet"

# Per-key history features in the order they appear in the lookup table.
HIST_FEATURES = {
    "card1": ("card1_hist_fraud_rate", "card1_hist_amt_mean", "card1_hist_count"),
    "addr1": ("addr1_hist_fraud_rate", "addr1_hist_amt_mean", "addr1_hist_count"),
}


class FeatureHistory:
    """In-memory lookup of global train aggregates, keyed by entity value."""

    def __init__(self, path: Path = HISTORY_PATH) -> None:
        self.maps: dict[str, dict[float, tuple[float, float, float]]] = {
            k: {} for k in HISTORY_KEYS
        }
        if path.exists():
            df = pd.read_parquet(path)
            for key in HISTORY_KEYS:
                fr, am, cnt = HIST_FEATURES[key]
                sub = df[df["key_type"] == key]
                for _, row in sub.iterrows():
                    try:
                        k = float(row["key"])
                    except (ValueError, TypeError):
                        continue
                    self.maps[key][k] = (
                        float(row[fr]),
                        float(row[am]),
                        float(row[cnt]),
                    )

    def has_data(self) -> bool:
        return any(self.maps.values())

    def lookup(self, key_type: str, value: float) -> tuple[float, float, float] | None:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return self.maps.get(key_type, {}).get(float(value))

    def recompute(
        self,
        features: dict[str, float],
        *,
        card1: float | None = None,
        addr1: float | None = None,
        amount: float | None = None,
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Return an updated feature vector plus a human-readable derivation log."""
        updated = dict(features)
        derivation: list[dict[str, Any]] = []

        for key_type, raw in (("card1", card1), ("addr1", addr1)):
            fr_c, am_c, cnt_c = HIST_FEATURES[key_type]
            hit = self.lookup(key_type, raw) if raw is not None else None
            if hit is not None:
                fraud_rate, amt_mean, count = hit
                updated[fr_c], updated[am_c], updated[cnt_c] = (
                    fraud_rate,
                    amt_mean,
                    count,
                )
                derivation.append(
                    {
                        "feature": fr_c,
                        "value": fraud_rate,
                        "note": (
                            f"{key_type}={raw}: found in training history "
                            f"({int(count)} past txns, fraud rate {fraud_rate:.4f})"
                        ),
                    }
                )
                derivation.append(
                    {
                        "feature": am_c,
                        "value": amt_mean,
                        "note": f"{key_type}={raw}: training mean amount {amt_mean:.2f}",
                    }
                )
                derivation.append(
                    {
                        "feature": cnt_c,
                        "value": count,
                        "note": f"{key_type}={raw}: training transaction count {int(count)}",
                    }
                )
            else:
                updated[fr_c], updated[am_c], updated[cnt_c] = 0.0, 0.0, 0.0
                derivation.append(
                    {
                        "feature": fr_c,
                        "value": 0.0,
                        "note": (
                            f"{key_type}={raw}: NOT seen in training → "
                            "new entity, history defaults to 0"
                        ),
                    }
                )

        if amount is not None:
            a = float(amount)
            updated["TransactionAmt"] = a
            updated["TransactionAmt_log"] = math.log1p(max(a, 0.0))
            updated["TransactionAmt_cents"] = (a - math.floor(a)) * 100.0
            derivation.append(
                {
                    "feature": "TransactionAmt",
                    "value": a,
                    "note": (
                        f"raw amount {a:.2f} → TransactionAmt / "
                        f"TransactionAmt_log / TransactionAmt_cents"
                    ),
                }
            )

        return updated, derivation
