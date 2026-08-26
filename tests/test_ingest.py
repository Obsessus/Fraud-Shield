"""Tests for the ingestion/join logic (run on synthetic data; no real dataset needed)."""

from __future__ import annotations

import pandas as pd
import pytest

from fraudintel.data.ingest import (
    ingest_split,
    join_transaction_identity,
    validate_transaction,
)


def _trans(ids):
    return pd.DataFrame(
        {
            "TransactionID": ids,
            "TransactionDT": [100 * i for i in ids],
            "isFraud": [0, 1, 0, 0, 1][: len(ids)],
        }
    )


def _ident(ids):
    return pd.DataFrame(
        {"TransactionID": ids, "DeviceType": ["desktop"] * len(ids), "id_12": [1] * len(ids)}
    )


def test_left_join_keeps_all_transactions():
    trans = _trans([1, 2, 3, 4, 5])
    ident = _ident([1, 3, 5])  # only some transactions have identity
    joined = join_transaction_identity(trans, ident)
    assert len(joined) == 5
    assert joined["TransactionID"].tolist() == [1, 2, 3, 4, 5]
    # identity cols NaN where no match
    assert joined.loc[joined.TransactionID == 2, "DeviceType"].isna().all()


def test_join_without_identity_returns_all_transactions():
    trans = _trans([1, 2, 3])
    joined = join_transaction_identity(trans, None)
    assert len(joined) == 3
    # no identity columns added
    assert "DeviceType" not in joined.columns


def test_validate_transaction_detects_duplicates():
    dup = pd.DataFrame({"TransactionID": [1, 1, 2], "isFraud": [0, 0, 1]})
    with pytest.raises(ValueError):
        validate_transaction(dup)


def test_validate_transaction_missing_key():
    with pytest.raises(ValueError):
        validate_transaction(pd.DataFrame({"x": [1]}))


def test_ingest_split_writes_parquet(tmp_path):
    raw = tmp_path / "raw"
    interim = tmp_path / "interim"
    raw.mkdir()
    _trans([10, 20, 30]).to_csv(raw / "train_transaction.csv", index=False)
    _ident([10, 30]).to_csv(raw / "train_identity.csv", index=False)

    df, out = ingest_split("train", raw=raw, interim=interim)
    assert out.exists()
    assert out.suffix == ".parquet"
    assert len(df) == 3
    reread = pd.read_parquet(out)
    assert len(reread) == 3
    assert "DeviceType" in reread.columns
