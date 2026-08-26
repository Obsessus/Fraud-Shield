"""Feature engineering for the Fraud Intelligence Platform.

All features are built to be leakage-free:
- Historical entity aggregates (fraud rate / amount stats) use ONLY past rows within
  the same entity, computed time-safely (the current row is excluded).
- Aggregate *mappings* are fit on the labeled TRAIN set and mapped onto TEST (which is
  a later, disjoint period), so test features never see train labels directly.
- Frequency encodings are fit on train and applied to test (counts are not target-based).

The raw context columns (C*, M*, V*, dist*) are passed through; downstream modeling
selects the final feature matrix.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

REFERENCE_DATE = "2017-11-30"
IDENTITY_KEYS = ["card1", "addr1", "P_emaildomain", "DeviceInfo"]
AGG_KEYS = ["card1", "addr1", "P_emaildomain"]


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Element-wise division that yields NaN where denominator is 0."""
    den = den.astype("float64")
    out = num.astype("float64") / den
    return out.where(den != 0, other=np.nan)


def add_temporal_features(df: pd.DataFrame, reference: str = REFERENCE_DATE) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_timedelta(df["TransactionDT"], unit="s") + pd.Timestamp(reference)
    df["DT_M"] = dt.dt.month.astype("int8")
    df["DT_W"] = dt.dt.isocalendar().week.astype("int").astype("int8")
    df["DT_D"] = dt.dt.day.astype("int8")
    df["hour"] = dt.dt.hour.astype("int8")
    df["day_of_week"] = dt.dt.dayofweek.astype("int8")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    return df


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    amt = df["TransactionAmt"].astype("float64")
    df["TransactionAmt_log"] = np.log1p(amt.clip(lower=0))
    df["TransactionAmt_cents"] = (amt - np.floor(amt)) * 100.0
    return df


def add_train_entity_features(
    df: pd.DataFrame,
    keys: list[str] | None = None,
    y_col: str = "isFraud",
    amt_col: str = "TransactionAmt",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Time-safe historical aggregates fit on train.

    For each entity key and each row, the historical fraud rate / amount mean use only
    rows that occurred strictly earlier (the current row is excluded). Also returns a
    global per-key mapping (from all train rows) used to featurize the future test set.
    """
    keys = keys or AGG_KEYS
    df = df.sort_values("TransactionDT").reset_index(drop=True).copy()
    mappings: dict[str, pd.DataFrame] = {}
    for key in keys:
        safe = f"__safe_{key}"
        df[safe] = df[key].fillna("__MISSING__")
        g = df.groupby(safe)
        prior_count = g.cumcount()  # rows before current, within key (time-ordered)
        prior_fraud = g[y_col].cumsum() - df[y_col].astype("float64")
        prior_amt = g[amt_col].cumsum() - df[amt_col].astype("float64")
        df[f"{key}_hist_count"] = prior_count.astype("int32")
        df[f"{key}_hist_fraud_rate"] = _safe_div(prior_fraud, prior_count)
        df[f"{key}_hist_amt_mean"] = _safe_div(prior_amt, prior_count)
        agg = (
            df.groupby(safe)
            .agg(
                _fraud_rate=(y_col, "mean"),
                _amt_mean=(amt_col, "mean"),
                _count=(y_col, "size"),
            )
            .reset_index()
            .rename(columns={safe: key})
            .set_index(key)
        )
        mappings[key] = agg
        df = df.drop(columns=[safe])
    return df, mappings


def add_test_entity_features(
    df: pd.DataFrame, mappings: dict[str, pd.DataFrame], keys: list[str] | None = None
) -> pd.DataFrame:
    """Map train-fitted global entity aggregates onto the (future) test set."""
    keys = keys or AGG_KEYS
    df = df.copy()
    for key in keys:
        agg = mappings[key]
        safe = df[key].fillna("__MISSING__")
        merged = safe.to_frame(name=key).merge(agg, left_on=key, right_index=True, how="left")
        df[f"{key}_hist_fraud_rate"] = merged["_fraud_rate"].values
        df[f"{key}_hist_amt_mean"] = merged["_amt_mean"].values
        df[f"{key}_hist_count"] = merged["_count"].fillna(0).values
    return df


def add_frequency_features(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    mappings: dict[str, dict[str, int]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    cols = cols or IDENTITY_KEYS
    df = df.copy()
    if mappings is None:
        mappings = {c: df[c].astype("string").value_counts().to_dict() for c in cols}
    for c in cols:
        df[f"{c}_freq"] = df[c].astype("string").map(mappings[c]).fillna(0).astype("int32")
    return df, mappings


def add_identity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    id_cols = [c for c in df.columns if c.startswith("id_")]
    if id_cols:
        df["n_identity_present"] = df[id_cols].notna().sum(axis=1).astype("int16")
    if "id_30" in df.columns:
        df["os_family"] = (
            df["id_30"].astype("string").str.extract(r"^([A-Za-z]+)", expand=False)
        )
    if "id_31" in df.columns:
        df["browser_family"] = df["id_31"].astype("string").str.split().str[0]
    for col in ("DeviceType", "DeviceInfo"):
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def build_train_features(train_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build all feature groups for train; return (frame, artifacts for test)."""
    df, ent_maps = add_train_entity_features(train_df)
    df, freq_maps = add_frequency_features(df, IDENTITY_KEYS)
    df = add_temporal_features(df)
    df = add_amount_features(df)
    df = add_identity_features(df)
    artifacts: dict[str, object] = {"entity": ent_maps, "freq": freq_maps}
    return df, artifacts


def build_test_features(test_df: pd.DataFrame, artifacts: dict[str, object]) -> pd.DataFrame:
    """Build the same feature groups for the future test set using train artifacts."""
    entity_maps = cast("dict[str, pd.DataFrame]", artifacts["entity"])
    freq_maps = cast("dict[str, dict[str, int]]", artifacts["freq"])
    df = add_test_entity_features(test_df, entity_maps)
    df, _ = add_frequency_features(df, IDENTITY_KEYS, mappings=freq_maps)
    df = add_temporal_features(df)
    df = add_amount_features(df)
    df = add_identity_features(df)
    return df


def build_temporal_splits(
    train_df: pd.DataFrame, t1: float, t2: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Leakage-safe features for a chronological 3-way split (expanding window).

    - Train split: aggregates fit on train rows only.
    - Validation split: maps train-only aggregates (no validation leakage).
    - Hold-out split: aggregates fit on train+validation (no hold-out leakage, and never
      touches the future Kaggle test set).
    Returns (train_feat, val_feat, holdout_feat), each retaining `isFraud`.
    """
    tr = train_df[train_df["TransactionDT"] <= t1].copy()
    va = train_df[(train_df["TransactionDT"] > t1) & (train_df["TransactionDT"] <= t2)].copy()
    ho = train_df[train_df["TransactionDT"] > t2].copy()

    tr_feat, tr_art = build_train_features(tr)
    va_feat = build_test_features(va, tr_art)

    comb, comb_art = build_train_features(pd.concat([tr, va], ignore_index=True))
    ho_feat = build_test_features(ho, comb_art)
    return tr_feat, va_feat, ho_feat
