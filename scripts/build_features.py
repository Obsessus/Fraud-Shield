"""Build features for train + test and write processed parquet via DVC stage.

Usage (or via `dvc repro`): python scripts/build_features.py
"""

from __future__ import annotations

import pandas as pd

from fraudintel.data.paths import interim_dir, processed_dir
from fraudintel.features.build import build_test_features, build_train_features


def main() -> None:
    train = pd.read_parquet(interim_dir() / "train_joined.parquet")
    test = pd.read_parquet(interim_dir() / "test_joined.parquet")

    tr_feat, artifacts = build_train_features(train)
    te_feat = build_test_features(test, artifacts)

    processed_dir().mkdir(parents=True, exist_ok=True)
    tr_feat.to_parquet(processed_dir() / "train_features.parquet")
    te_feat.to_parquet(processed_dir() / "test_features.parquet")

    print(f"[features] train: {tr_feat.shape} -> data/processed/train_features.parquet")
    print(f"[features] test:  {te_feat.shape} -> data/processed/test_features.parquet")
    feat_markers = ("hist_", "_freq", "DT_", "TransactionAmt_", "os_", "browser_", "n_identity")
    new_cols = [c for c in tr_feat.columns if any(k in c for k in feat_markers)]
    print(f"[features] new feature cols: {new_cols}")


if __name__ == "__main__":
    main()
