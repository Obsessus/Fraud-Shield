"""Validate the joined train/test frames and write a JSON report to artifacts/."""

from __future__ import annotations

import json

import pandas as pd

from fraudintel.data.paths import artifacts_dir, interim_dir
from fraudintel.data.validate import validate


def main() -> None:
    artifacts_dir().mkdir(parents=True, exist_ok=True)
    report = {}
    for stage in ("train", "test"):
        df = pd.read_parquet(interim_dir() / f"{stage}_joined.parquet")
        res = validate(stage, df)
        report[stage] = {"stats": res.stats, "warnings": res.warnings, "errors": res.errors}
        print(f"[validate] {stage}: ok={res.ok} rows={res.stats['rows']} "
              f"fraud_rate={res.stats.get('fraud_rate')} "
              f"identity_coverage={res.stats.get('identity_coverage')}")

    out = artifacts_dir() / "validation_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"[validate] report -> {out}")


if __name__ == "__main__":
    main()
