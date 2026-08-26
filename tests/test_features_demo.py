"""Unit tests for the real-time feature derivation used by the UI demo."""

from __future__ import annotations

import pandas as pd
import pytest

from fraudintel.serving.features_demo import FeatureHistory


def _make_history(tmp_path) -> FeatureHistory:
    df = pd.DataFrame(
        [
            # key_type, key, fraud_rate, amt_mean, count
            ("card1", 1000.0, 0.10, 150.0, 40.0),
            ("card1", 2000.0, 0.02, 90.0, 120.0),
            ("addr1", 300.0, 0.05, 120.0, 200.0),
        ],
        columns=[
            "key_type", "key",
            "card1_hist_fraud_rate", "card1_hist_amt_mean", "card1_hist_count",
        ],
    )
    # addr1 columns are required by the reader even if only card1 rows exist here;
    # add them so the parquet schema is complete.
    df["addr1_hist_fraud_rate"] = 0.0
    df["addr1_hist_amt_mean"] = 0.0
    df["addr1_hist_count"] = 0.0
    path = tmp_path / "feature_history.parquet"
    df.to_parquet(path, index=False)
    return FeatureHistory(path)


def test_lookup_known_card1(tmp_path):
    hist = _make_history(tmp_path)
    feats, deriv = hist.recompute({}, card1=1000.0)
    assert feats["card1_hist_fraud_rate"] == 0.10
    assert feats["card1_hist_amt_mean"] == 150.0
    assert feats["card1_hist_count"] == 40.0
    assert any("found in training history" in d["note"] for d in deriv)


def test_unknown_card1_defaults_to_zero(tmp_path):
    hist = _make_history(tmp_path)
    feats, deriv = hist.recompute({}, card1=9999.0)
    assert feats["card1_hist_fraud_rate"] == 0.0
    assert feats["card1_hist_amt_mean"] == 0.0
    assert feats["card1_hist_count"] == 0.0
    assert any("NOT seen in training" in d["note"] for d in deriv)


def test_amount_features_recomputed(tmp_path):
    hist = _make_history(tmp_path)
    feats, deriv = hist.recompute({}, amount=250.75)
    assert feats["TransactionAmt"] == 250.75
    assert feats["TransactionAmt_log"] == pytest.approx(__import__("math").log1p(250.75))
    assert feats["TransactionAmt_cents"] == pytest.approx(75.0)
    assert any(d["feature"] == "TransactionAmt" for d in deriv)


def test_existing_features_preserved_except_history(tmp_path):
    hist = _make_history(tmp_path)
    base = {"TransactionAmt": 10.0, "C1": 3.0, "card1_hist_fraud_rate": 0.99}
    feats, _ = hist.recompute(base, card1=1000.0)
    assert feats["C1"] == 3.0  # untouched
    assert feats["card1_hist_fraud_rate"] == 0.10  # overwritten from history
