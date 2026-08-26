"""Build a small, UI-friendly sample of real transactions (Stage 19).

The serving container is memory-limited, so instead of letting the UI stream a row
from the full ~590k-row `train_features.parquet` at request time, we pre-materialize
a tiny, representative parquet of real transactions (all 416 engineered features +
the `isFraud` label). The `/sample` endpoint reads only this small file.
"""

from __future__ import annotations

import pandas as pd

from fraudintel.data.paths import artifacts_dir, processed_dir

SAMPLE_ROWS = 300
SEED = 7


def main() -> None:
    src = processed_dir() / "train_features.parquet"
    df = pd.read_parquet(src)
    # Keep only numeric columns (the model is trained on numeric features). The
    # `isFraud` label is numeric too, so it stays as the ground-truth label.
    # Coerce non-finite values so the JSON the UI consumes is always valid.
    numeric = (
        df.select_dtypes(include="number")
        .replace([float("inf"), float("-inf")], 0.0)
        .fillna(0.0)
    )
    sample = numeric.sample(n=min(SAMPLE_ROWS, len(numeric)), random_state=SEED).copy()

    out_dir = artifacts_dir() / "ui"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sample_features.parquet"
    sample.to_parquet(out_path, index=False)
    print(f"[ui-sample] wrote {out_path} ({len(sample)} rows, {sample.shape[1]} cols)")


if __name__ == "__main__":
    main()
