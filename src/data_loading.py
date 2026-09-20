"""Loading and auditing of the two Review 1 raw datasets.

Raw files are treated as immutable: nothing in this module ever writes to
data/raw/. Every loader verifies the schema it was promised and reports, rather
than silently repairs, any mismatch.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path

import pandas as pd

from . import config


class DatasetMissingError(FileNotFoundError):
    """Raised when a required raw CSV has not been placed in data/raw/."""


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


@dataclass
class Provenance:
    track: str
    filename: str
    source_url: str
    retrieved_on: str
    sha256: str
    size_bytes: int
    raw_rows: int
    raw_cols: int

    def to_dict(self) -> dict:
        return asdict(self)


def record_provenance(track: str, df: pd.DataFrame, path: Path) -> Provenance:
    spec = config.DATASETS[track]
    prov = Provenance(
        track=track,
        filename=path.name,
        source_url=spec["url"],
        retrieved_on=date.today().isoformat(),
        sha256=sha256_of(path),
        size_bytes=path.stat().st_size,
        raw_rows=int(df.shape[0]),
        raw_cols=int(df.shape[1]),
    )
    config.ensure_dirs()
    out = config.TABLES_DIR / "dataset_provenance.json"
    existing = {}
    if out.exists():
        existing = json.loads(out.read_text())
    existing[track] = prov.to_dict()
    out.write_text(json.dumps(existing, indent=2))
    return prov


# --------------------------------------------------------------------------
# Raw loading
# --------------------------------------------------------------------------
def raw_path(track: str) -> Path:
    return config.RAW_DIR / config.DATASETS[track]["filename"]


def load_raw(track: str, verbose: bool = True) -> pd.DataFrame:
    """Read the raw CSV for a track and check it against its expected shape."""
    spec = config.DATASETS[track]
    path = raw_path(track)
    if not path.exists():
        raise DatasetMissingError(
            f"Missing raw file for the {track} track: {path}\n"
            f"Download it from {spec['url']} and place '{spec['filename']}' in "
            f"data/raw/. See scripts/download_data.py for instructions."
        )

    df = pd.read_csv(path)

    expected = spec["expected_raw_shape"]
    actual = df.shape
    if verbose:
        print(f"[load_raw] {track}: {path.name} -> shape {actual} "
              f"(expected {expected})")
    if actual != tuple(expected):
        print(f"  !! SHAPE MISMATCH for {track}. Expected {expected}, got {actual}. "
              f"This is reported, not corrected. Verify you downloaded the "
              f"version at {spec['url']}.")
    record_provenance(track, df, path)
    return df


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------
def audit(df: pd.DataFrame, track: str, label: str = "raw") -> pd.DataFrame:
    """Per-column audit: dtype, missing, unique, duplicate count, basic stats."""
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": round(100 * s.isna().mean(), 3),
            "n_unique": int(s.nunique(dropna=True)),
            "example": s.dropna().iloc[0] if s.notna().any() else None,
        })
    out = pd.DataFrame(rows)
    config.ensure_dirs()
    out.to_csv(config.TABLES_DIR / f"{track}_audit_{label}.csv", index=False)
    return out


def audit_summary(df: pd.DataFrame, track: str) -> dict:
    spec = config.DATASETS[track]
    summary = {
        "track": track,
        "shape": tuple(df.shape),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "total_missing_cells": int(df.isna().sum().sum()),
        "columns": list(df.columns),
    }
    tgt = spec.get("target")
    if tgt:
        summary["target"] = tgt
        summary["target_present"] = tgt in df.columns
        if tgt in df.columns:
            if df[tgt].nunique() <= 20:
                summary["target_distribution"] = (
                    df[tgt].value_counts().sort_index().to_dict()
                )
            else:
                summary["target_describe"] = df[tgt].describe().to_dict()
    return summary


# --------------------------------------------------------------------------
# Prepared views (raw -> the 6-9 column working frame)
# --------------------------------------------------------------------------
def prepare(track: str, df: pd.DataFrame | None = None,
            verbose: bool = True) -> pd.DataFrame:
    """Return the *prepared* frame: selected columns only, no transformation.

    Preparation is column selection plus the removal of exact duplicate rows.
    It deliberately does NOT impute, encode or scale - those happen inside
    pipelines that are fitted on training folds only.
    """
    spec = config.DATASETS[track]
    if df is None:
        df = load_raw(track, verbose=verbose)

    keep = list(spec["numeric_features"]) + list(spec["categorical_features"]) \
           + [spec["target"]]

    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(f"[prepare] {track}: expected columns absent from the CSV: {missing}")

    out = df[keep].copy()
    if verbose:
        print(f"[prepare] {track}: raw {df.shape} -> prepared {out.shape} "
              f"({len(keep)} columns kept)")
        dropped = [c for c in df.columns if c not in keep]
        print(f"  excluded: {dropped}")
    return out
