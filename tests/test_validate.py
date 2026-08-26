"""Tests for data validation (synthetic frames; no real dataset needed)."""

from __future__ import annotations

import pandas as pd
import pytest

from fraudintel.data.validate import validate


def _train(extra=None):
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [100, 200, 300],
            "TransactionAmt": [10.0, 20.0, 30.0],
            "isFraud": [0, 1, 0],
            "DeviceType": ["desktop", None, "desktop"],
        }
    )
    if extra:
        df = pd.concat([df, extra], axis=1)
    return df


def test_valid_train_passes():
    rep = validate("train", _train())
    assert rep.ok
    assert rep.stats["fraud_rate"] == pytest.approx(1 / 3)
    assert rep.stats["identity_coverage"] == pytest.approx(2 / 3, abs=1e-3)


def test_duplicate_transaction_id_fails():
    df = _train()
    df.loc[0, "TransactionID"] = 2  # now duplicated
    with pytest.raises(ValueError):
        validate("train", df)


def test_non_binary_target_fails():
    df = _train()
    df.loc[0, "isFraud"] = 2
    with pytest.raises(ValueError):
        validate("train", df)


def test_nonpositive_amount_fails():
    df = _train()
    df.loc[0, "TransactionAmt"] = 0.0
    with pytest.raises(ValueError):
        validate("train", df)


def test_missing_target_fails():
    df = _train().drop(columns=["isFraud"])
    with pytest.raises(ValueError):
        validate("train", df)


def test_valid_test_passes_without_target():
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "TransactionDT": [100, 200],
            "TransactionAmt": [10.0, 20.0],
            "DeviceType": [None, "mobile"],
        }
    )
    rep = validate("test", df)
    assert rep.ok


def test_test_missing_base_column_fails():
    df = pd.DataFrame({"TransactionID": [1], "TransactionDT": [100]})
    with pytest.raises(ValueError):
        validate("test", df)
