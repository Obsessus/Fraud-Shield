"""Temporal (chronological) train/validation/hold-out split.

Fraud is time-dependent, so we never split randomly. We order by ``TransactionDT``
(seconds from a reference datetime) and cut at time thresholds so each split is a
contiguous period. The Kaggle *test* set is a later, disjoint period and is treated
as the future hold-out for production simulation (no labels).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraudintel.data.paths import splits_dir


def to_datetime(transaction_dt: pd.Series, reference: str = "2017-11-30") -> pd.Series:
    """Convert ``TransactionDT`` (seconds from reference) to a real datetime."""
    ref = pd.Timestamp(reference)
    return ref + pd.to_timedelta(transaction_dt, unit="s")


def compute_boundaries(
    transaction_dt: pd.Series, train_frac: float, val_frac: float
) -> tuple[float, float]:
    """Return ``(t1, t2)`` TransactionDT thresholds splitting time into 3 periods.

    train: dt <= t1 ; validation: t1 < dt <= t2 ; hold-out: dt > t2.
    """
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError("need 0 < train_frac, val_frac and train_frac+val_frac < 1")
    dt = transaction_dt.astype(float)
    t1 = float(np.quantile(dt, train_frac))
    t2 = float(np.quantile(dt, train_frac + val_frac))
    return t1, t2


def assign_split(
    transaction_dt: pd.Series, t1: float, t2: float
) -> pd.Series:
    """Label each row as train / validation / holdout by time threshold."""
    dt = transaction_dt.astype(float)
    return pd.cut(
        dt,
        bins=[-np.inf, t1, t2, np.inf],
        labels=["train", "validation", "holdout"],
        include_lowest=True,
    )


def run_split(
    train_df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    reference: str = "2017-11-30",
    out_dir: Path | None = None,
) -> dict[str, object]:
    """Compute the temporal split on the labeled train set and write a manifest.

    Returns a summary dict (rows, fraud rate, date range per split).
    """
    out_dir = Path(out_dir) if out_dir is not None else splits_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    t1, t2 = compute_boundaries(train_df["TransactionDT"], train_frac, val_frac)
    split = assign_split(train_df["TransactionDT"], t1, t2)
    summary = _summarize(train_df, split, t1, t2, reference)

    manifest = {
        "reference_date": reference,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "thresholds_seconds": {"t1": t1, "t2": t2},
        "summary": summary,
    }
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _summarize(
    train_df: pd.DataFrame,
    split: pd.Series,
    t1: float,
    t2: float,
    reference: str,
) -> dict[str, object]:
    out: dict[str, object] = {}
    dt = to_datetime(train_df["TransactionDT"], reference)
    for name in ("train", "validation", "holdout"):
        mask = split == name
        sub = train_df[mask]
        out[name] = {
            "rows": int(mask.sum()),
            "fraud_rate": round(float(sub["isFraud"].mean()), 6) if len(sub) else None,
            "n_positives": int(sub["isFraud"].sum()) if len(sub) else 0,
            "date_min": str(dt[mask].min()),
            "date_max": str(dt[mask].max()),
        }
    return out
