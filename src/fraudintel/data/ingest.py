"""Ingestion: load raw IEEE-CIS CSVs and join transaction + identity.

Design rules:
- ``TransactionID`` is the join key and must be unique on the transaction side.
- Identity is joined with a LEFT join so transactions without identity rows are
  preserved (identity is missing for a subset of transactions at inference too).
- Functions accept DataFrames where possible so they are unit-testable without
  the real (large) dataset.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import interim_dir, raw_dir


def load_raw_csv(path: str | Path) -> pd.DataFrame:
    """Load a raw CSV with consistent parsing (avoids mixed-type warnings)."""
    return pd.read_csv(path, low_memory=False)


def validate_transaction(transaction: pd.DataFrame) -> None:
    """Validate the transaction frame before joining.

    Raises:
        ValueError: if ``TransactionID`` is missing or not unique.
    """
    if "TransactionID" not in transaction.columns:
        raise ValueError("transaction frame missing 'TransactionID'")
    if transaction["TransactionID"].duplicated().any():
        raise ValueError("duplicate TransactionID in transaction frame (join key must be unique)")


def join_transaction_identity(
    transaction: pd.DataFrame, identity: pd.DataFrame | None
) -> pd.DataFrame:
    """Left-join identity onto transaction on ``TransactionID``.

    Transactions without a matching identity row are retained (identity columns
    will be NaN for them), mirroring production where identity is often absent.
    """
    validate_transaction(transaction)
    if identity is None or identity.empty:
        return transaction.copy()
    if "TransactionID" not in identity.columns:
        raise ValueError("identity frame missing 'TransactionID'")
    return transaction.merge(identity, on="TransactionID", how="left")


def ingest_split(
    split: str, raw: Path | None = None, interim: Path | None = None
) -> tuple[pd.DataFrame, Path]:
    """Load and join one split ('train' or 'test') and write an interim parquet.

    Returns the joined DataFrame and the output parquet path.
    """
    raw = Path(raw) if raw is not None else raw_dir()
    interim = Path(interim) if interim is not None else interim_dir()

    t_path = raw / f"{split}_transaction.csv"
    i_path = raw / f"{split}_identity.csv"
    if not t_path.exists():
        raise FileNotFoundError(f"missing {t_path}")

    transaction = load_raw_csv(t_path)
    identity = load_raw_csv(i_path) if i_path.exists() else None
    joined = join_transaction_identity(transaction, identity)

    interim.mkdir(parents=True, exist_ok=True)
    out = interim / f"{split}_joined.parquet"
    joined.to_parquet(out, index=False)
    return joined, out
