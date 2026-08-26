"""Download the IEEE-CIS Fraud Detection dataset from Kaggle into ``data/raw``.

Requires Kaggle credentials, provided either as:
  - a ``~/.kaggle/kaggle.json`` file, or
  - environment variables ``KAGGLE_USERNAME`` and ``KAGGLE_KEY``.

The downloaded zip is extracted in place. Nothing in this script is committed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

COMPETITION = "ieee-fraud-detection"


def main() -> int:
    raw = Path(os.environ.get("FIP_DATA_DIR", Path(__file__).resolve().parents[1] / "data")) / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    print(f"[download] fetching '{COMPETITION}' -> {raw}")
    # Locate the kaggle console script next to the running interpreter (PATH-independent).
    bin_dir = Path(sys.executable).parent
    kaggle_exe = bin_dir / "kaggle.exe"
    if not kaggle_exe.exists():
        kaggle_exe = bin_dir / "kaggle"
    cmd = [str(kaggle_exe), "competitions", "download", "-c", COMPETITION, "-p", str(raw)]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("[download] ERROR: 'kaggle' not found. Install: pip install 'kaggle==1.6.17'")
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"[download] ERROR: kaggle download failed ({exc}). Check credentials.")
        return 1

    for z in raw.glob("*.zip"):
        print(f"[download] extracting {z.name}")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(raw)

    print("[download] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
