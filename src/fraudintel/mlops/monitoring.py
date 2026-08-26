"""Data-drift monitoring logic (Stage 17).

Compares a reference feature distribution against a current batch using Evidently's
``DataDriftPreset`` and returns a compact summary. Kept in ``src/`` so it is unit-testable
and reusable; ``scripts/drift_check.py`` is the runnable entrypoint.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report


def run_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
    """Run an Evidently data-drift report; return a compact summary dict."""
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    result = report.as_dict()

    dataset_drift = False
    drifted = 0
    total = 0
    details: list[dict[str, Any]] = []
    for m in result.get("metrics", []):
        if not isinstance(m, dict):
            continue
        res = m.get("result", {})
        if m.get("metric") == "DataDriftTable":
            dataset_drift = bool(res.get("dataset_drift", False))
            total = int(res.get("number_of_columns", 0))
            drifted = int(res.get("number_of_drifted_columns", 0))
            for col, info in res.get("drift_by_columns", {}).items():
                details.append(
                    {"column": col, "drift_detected": bool(info.get("drift_detected", False))}
                )
            break
        if m.get("metric") == "DatasetDriftMetric":
            dataset_drift = bool(res.get("dataset_drift", False))
            total = int(res.get("number_of_columns", total))
            drifted = int(res.get("number_of_drifted_columns", drifted))
        elif m.get("metric") == "ColumnDriftMetric" and "column_name" in res:
            details.append(
                {
                    "column": res["column_name"],
                    "drift_detected": bool(res.get("drift_detected", False)),
                }
            )

    return {
        "dataset_drift": dataset_drift,
        "drifted_columns": drifted,
        "total_columns": total,
        "details": details,
    }
