"""Data layer: paths, ingestion, and join logic for the Fraud Intelligence Platform.

Production data code lives here. Notebooks remain exploration-only.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def data_root() -> Path:
    """Root data directory; override with ``FIP_DATA_DIR`` if needed."""
    import os

    return Path(os.environ.get("FIP_DATA_DIR", PROJECT_ROOT / "data"))


def raw_dir() -> Path:
    return data_root() / "raw"


def interim_dir() -> Path:
    return data_root() / "interim"


def artifacts_dir() -> Path:
    return data_root() / "artifacts"


def processed_dir() -> Path:
    return data_root() / "processed"


def splits_dir() -> Path:
    return data_root() / "splits"
