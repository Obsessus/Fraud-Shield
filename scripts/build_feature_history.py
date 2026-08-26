"""Build the real-time feature-history index used by the UI demo.

The model consumes leakage-safe *historical aggregate* features such as
``card1_hist_fraud_rate`` / ``card1_hist_amt_mean`` / ``card1_hist_count``. These
are global train aggregates for an entity key (card1 / addr1), mapped onto every
row that shares that key. This script extracts that mapping from the engineered
training frame into a small lookup table so the UI can recompute those features
"in real time" for a raw transaction (exactly mirroring the offline logic in
``features/build.py``: ``add_test_entity_features``).
"""

from __future__ import annotations

import pandas as pd

from fraudintel.data.paths import artifacts_dir, processed_dir

HISTORY_KEYS = ("card1", "addr1")


def main() -> None:
    src = processed_dir() / "train_features.parquet"
    df = pd.read_parquet(src)

    frames = []
    for key in HISTORY_KEYS:
        fr_c, am_c, cnt_c = (
            f"{key}_hist_fraud_rate",
            f"{key}_hist_amt_mean",
            f"{key}_hist_count",
        )
        # Mirror the offline mapping in `features/build.py::add_train_entity_features`:
        # the history feature for an entity key is the GLOBAL train aggregate —
        # mean fraud rate, mean amount, and the number of training transactions.
        sub = (
            df.groupby(key)
            .agg(
                **{fr_c: ("isFraud", "mean")},
                **{am_c: ("TransactionAmt", "mean")},
                **{cnt_c: ("isFraud", "size")},
            )
            .reset_index()
            .rename(columns={key: "key"})
        )
        sub["key_type"] = key
        sub = sub[["key_type", "key", fr_c, am_c, cnt_c]]
        frames.append(sub)

    out = pd.concat(frames, ignore_index=True).fillna(0.0)

    out_dir = artifacts_dir() / "ui"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "feature_history.parquet"
    out.to_parquet(out_path, index=False)
    print(
        f"[feature-history] wrote {out_path} "
        f"({len(out)} keys across {len(HISTORY_KEYS)} entity types)"
    )


if __name__ == "__main__":
    main()
