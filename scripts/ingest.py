"""DVC/runtime entrypoint: join raw transaction+identity for train and test splits.

Run directly or via ``dvc repro``. Missing raw files are reported, not silently ignored.
"""

from __future__ import annotations

from fraudintel.data.ingest import ingest_split

if __name__ == "__main__":
    for split in ("train", "test"):
        try:
            df, out = ingest_split(split)
            print(f"[ingest] {split}: {len(df):,} rows -> {out}")
        except FileNotFoundError as exc:
            print(f"[ingest] SKIP {split}: {exc}")
