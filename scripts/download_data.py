"""Dataset acquisition helper.

Order of operations:
  1. If the CSV is already in data/raw/, verify and stop.
  2. Otherwise try the Kaggle API (needs ~/.kaggle/kaggle.json).
  3. Otherwise print exact manual instructions.

This script NEVER asks for credentials in source code, never writes secrets to
the repository, and never substitutes a different dataset.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data_loading import sha256_of


def manual_instructions(spec: dict) -> str:
    return f"""
  MANUAL DOWNLOAD - {spec['name']}
  ---------------------------------------------------------------
  1. Open: {spec['url']}
  2. Sign in to Kaggle and click "Download" (top right).
  3. Unzip the archive.
  4. Copy '{spec['filename']}' into: {config.RAW_DIR}
  Expected raw shape: {spec['expected_raw_shape'][0]} rows x {spec['expected_raw_shape'][1]} columns

  Kaggle API alternative (no credentials in code):
    pip install kaggle
    # Kaggle -> Account -> "Create New API Token" -> kaggle.json
    # place it at ~/.kaggle/kaggle.json  (Windows: %USERPROFILE%\\.kaggle\\kaggle.json)
    chmod 600 ~/.kaggle/kaggle.json
    kaggle datasets download -d {spec['kaggle_ref']} -p data/raw --unzip
"""


def try_kaggle(spec: dict) -> bool:
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except Exception:
        print("  kaggle CLI not available.")
        return False
    print(f"  attempting: kaggle datasets download -d {spec['kaggle_ref']}")
    r = subprocess.run(
        ["kaggle", "datasets", "download", "-d", spec["kaggle_ref"],
         "-p", str(config.RAW_DIR), "--unzip"],
        capture_output=True, text=True)
    if r.returncode != 0:
        print("  kaggle download failed:")
        print("  " + (r.stderr or r.stdout).strip().replace("\n", "\n  "))
        return False
    return (config.RAW_DIR / spec["filename"]).exists()


def main() -> int:
    config.ensure_dirs()
    missing = []
    for track, spec in config.DATASETS.items():
        path = config.RAW_DIR / spec["filename"]
        print(f"\n[{track}] {spec['filename']}")
        if path.exists():
            print(f"  present ({path.stat().st_size:,} bytes)")
            print(f"  sha256 {sha256_of(path)}")
            continue
        print("  not found in data/raw/")
        if try_kaggle(spec) and path.exists():
            print(f"  downloaded ({path.stat().st_size:,} bytes)")
            print(f"  sha256 {sha256_of(path)}")
            continue
        missing.append(spec)

    if missing:
        print("\n" + "=" * 70)
        print("ACTION REQUIRED - the following datasets must be supplied manually")
        print("=" * 70)
        for spec in missing:
            print(manual_instructions(spec))
        return 1
    print("\nAll three datasets are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
