"""Splitting and preprocessing.

Design rule enforced throughout: every transformer lives *inside* a
scikit-learn Pipeline, so it is refitted from scratch on the training portion
of each cross-validation fold. Nothing is ever fitted on the full dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def split_supervised(df: pd.DataFrame, target: str, stratify: bool = False,
                     stratify_bins: int | None = None,
                     test_size: float = config.TEST_SIZE,
                     random_state: int = config.RANDOM_STATE):
    """80:20 split used by every model within a track.

    stratify=True          -> stratify on the raw target (classification).
    stratify_bins=k        -> stratify on quantile bins of a continuous target.
                              Off by default; see docs/instructor_clarifications.md
                              item IC-2 for why regression is not blindly stratified.
    """
    X = df.drop(columns=[target])
    y = df[target]

    strat = None
    if stratify:
        strat = y
    elif stratify_bins:
        strat = pd.qcut(y, q=stratify_bins, labels=False, duplicates="drop")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    print(f"[split] train={X_train.shape} test={X_test.shape} "
          f"stratify={'target' if stratify else (f'{stratify_bins} quantile bins' if stratify_bins else 'none')}")
    return X_train, X_test, y_train, y_test


def check_no_row_leakage(X_train: pd.DataFrame, X_test: pd.DataFrame) -> dict:
    """Count exact-duplicate feature rows shared by train and test.

    Reported, not silently removed: with synthetic low-cardinality tabular data
    a handful of coincidental repeats is expected and is not the same thing as
    entity leakage.
    """
    tr = set(map(tuple, X_train.to_numpy()))
    te = list(map(tuple, X_test.to_numpy()))
    shared = sum(1 for r in te if r in tr)
    return {"test_rows": len(te), "test_rows_also_in_train": shared,
            "pct": round(100 * shared / max(len(te), 1), 3)}


def drop_exact_duplicates(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    n0 = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    if verbose:
        print(f"[dedup] {n0} -> {len(out)} rows ({n0 - len(out)} exact duplicates removed)")
    return out


# --------------------------------------------------------------------------
# Column transformers
# --------------------------------------------------------------------------
def make_preprocessor(numeric: list[str], categorical: list[str],
                      scale: bool = True) -> ColumnTransformer:
    """Median-impute + (optionally) standardise numerics; most-frequent-impute +
    one-hot encode categoricals.

    scale=False is used for tree/ensemble models, where standardisation is a
    no-op on the split points but costs interpretability of the raw units.
    """
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))

    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary",
                                 sparse_output=False)),
    ]

    transformers = []
    if numeric:
        transformers.append(("num", Pipeline(num_steps), numeric))
    if categorical:
        transformers.append(("cat", Pipeline(cat_steps), categorical))

    return ColumnTransformer(transformers, remainder="drop",
                             verbose_feature_names_out=False)


def feature_names_from(preprocessor: ColumnTransformer) -> list[str]:
    """Readable output feature names for coefficient / importance plots."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []
