"""Data-drift monitoring for the Fraud Intelligence Platform (Stage 17).

Offline drift check: compares a reference feature distribution (training features) against a
current batch (held-out features by default) using Evidently's ``DataDriftPreset`` and writes
a JSON report. Intended to run periodically (CI/cron) — not per-request.

Usage: python scripts/drift_check.py
"""

from __future__ import annotations

import json

import pandas as pd

from fraudintel.data.paths import artifacts_dir, processed_dir
from fraudintel.mlops.monitoring import run_drift

SAMPLE_ROWS = 20000


def main() -> None:
    ref = pd.read_parquet(processed_dir() / "train_features.parquet")
    cur = pd.read_parquet(processed_dir() / "test_features.parquet")

    # Align to numeric columns common to both sets (the parquet outputs can
    # differ in a few derived columns); drop the target from the reference.
    target = "isFraud"
    common = [c for c in ref.columns if c in cur.columns and c != target]
    ref = ref[common].select_dtypes(include="number")
    cur = cur[common].select_dtypes(include="number")
    cols = [c for c in ref.columns if c in cur.columns]
    ref = ref[cols]
    cur = cur[cols]

    ref = ref.sample(n=min(SAMPLE_ROWS, len(ref)), random_state=42)
    cur = cur.sample(n=min(SAMPLE_ROWS, len(cur)), random_state=42)

    summary = run_drift(ref, cur)
    out = artifacts_dir() / "monitoring" / "drift_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(
        f"[drift] dataset_drift={summary['dataset_drift']} "
        f"drifted={summary['drifted_columns']}/{summary['total_columns']}"
    )
    print(f"[drift] -> {out}")


if __name__ == "__main__":
    main()
