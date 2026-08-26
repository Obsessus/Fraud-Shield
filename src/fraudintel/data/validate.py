"""Data validation for the Fraud Intelligence Platform.

Validation is intentionally strict on structural integrity (join key, target, types)
and soft on distribution (warnings only). It runs on the joined frames so it works
for both train and test splits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

REQUIRED_BASE = ["TransactionID", "TransactionDT", "TransactionAmt"]
REQUIRED_TRAIN = REQUIRED_BASE + ["isFraud"]


@dataclass
class ValidationReport:
    stage: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_errors(self) -> None:
        if self.errors:
            raise ValueError(f"validation failed [{self.stage}]: " + "; ".join(self.errors))


def validate(stage: str, df: pd.DataFrame) -> ValidationReport:
    """Validate a joined transaction+identity frame.

    Args:
        stage: ``"train"`` (requires target) or ``"test"``.
        df: joined frame.

    Returns:
        ValidationReport with errors (hard failures), warnings, and basic stats.

    Raises:
        ValueError: if any hard error is found.
    """
    if stage not in ("train", "test"):
        raise ValueError(f"unknown stage: {stage}")
    rep = ValidationReport(stage=stage)
    required = REQUIRED_TRAIN if stage == "train" else REQUIRED_BASE

    missing = [c for c in required if c not in df.columns]
    if missing:
        rep.errors.append(f"missing required columns: {missing}")

    if "TransactionID" in df.columns and df["TransactionID"].duplicated().any():
        rep.errors.append("duplicate TransactionID (join key must be unique)")

    if "TransactionDT" in df.columns and df["TransactionDT"].min() < 0:
        rep.errors.append("negative TransactionDT")

    if "TransactionAmt" in df.columns:
        n_nonpos = int((df["TransactionAmt"] <= 0).sum())
        if n_nonpos:
            rep.errors.append(f"{n_nonpos} non-positive TransactionAmt")

    if stage == "train":
        if "isFraud" in df.columns:
            if not df["isFraud"].isin([0, 1]).all():
                rep.errors.append("isFraud contains values other than {0,1}")
            rep.stats["fraud_rate"] = round(float(df["isFraud"].mean()), 6)
            rep.stats["n_positives"] = int(df["isFraud"].sum())
        else:
            rep.errors.append("train frame missing isFraud target")

    if "DeviceType" in df.columns:
        cov = float(df["DeviceType"].notna().mean())
        rep.stats["identity_coverage"] = round(cov, 4)
        if cov < 0.1:
            rep.warnings.append(f"very low identity coverage ({cov:.2%})")

    rep.stats["rows"] = len(df)
    rep.stats["n_columns"] = df.shape[1]
    rep.raise_if_errors()
    return rep
