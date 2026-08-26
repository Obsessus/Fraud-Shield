"""Tests for the temporal split (synthetic data; no real dataset needed)."""

from __future__ import annotations

import pandas as pd

from fraudintel.data.split import assign_split, compute_boundaries, run_split


def _df():
    return pd.DataFrame(
        {
            "TransactionID": list(range(10)),
            "TransactionDT": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
            "isFraud": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        }
    )


def test_boundaries_monotonic():
    t1, t2 = compute_boundaries(_df()["TransactionDT"], 0.6, 0.3)
    assert t1 < t2


def test_assign_split_contiguous_nonoverlapping():
    df = _df()
    t1, t2 = compute_boundaries(df["TransactionDT"], 0.6, 0.3)
    split = assign_split(df["TransactionDT"], t1, t2)
    counts = split.value_counts().to_dict()
    # train: dt<=~54 -> 6 rows; validation: ~54<dt<=~81 -> 3; holdout: 1
    assert counts == {"train": 6, "validation": 3, "holdout": 1}
    assert set(split.unique()) == {"train", "validation", "holdout"}


def test_run_split_writes_manifest(tmp_path):
    manifest = run_split(_df(), train_frac=0.6, val_frac=0.3, out_dir=tmp_path)
    assert (tmp_path / "split_manifest.json").exists()
    assert set(manifest["summary"].keys()) == {"train", "validation", "holdout"}
    assert manifest["summary"]["train"]["rows"] == 6
    assert manifest["summary"]["holdout"]["rows"] == 1
