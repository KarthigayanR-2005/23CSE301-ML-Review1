"""Metric computation and comparison tables.

All metrics are computed on the SAME held-out split within a track, so the
comparison tables required by rubric items C2 (regression) and A2
(classification) are fair.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    precision_score, r2_score, recall_score, roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import cross_val_score

from . import config


# --------------------------------------------------------------------------
# Regression
# --------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict:
    """R2 / RMSE / MAE in the ORIGINAL target units (the target is never scaled)."""
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def evaluate_regressors(models: dict, X_train, y_train, X_test, y_test,
                        verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    rows, fitted = [], {}
    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        m_test = regression_metrics(y_test, model.predict(X_test))
        m_train = regression_metrics(y_train, model.predict(X_train))
        rows.append({
            "Model": name,
            "R2": m_test["R2"], "RMSE": m_test["RMSE"], "MAE": m_test["MAE"],
            "Train_R2": m_train["R2"], "Fit_seconds": round(fit_s, 3),
        })
        fitted[name] = model
        if verbose:
            print(f"  {name:<34} testR2={m_test['R2']:.4f} "
                  f"RMSE={m_test['RMSE']:.4f} MAE={m_test['MAE']:.4f}")
    table = (pd.DataFrame(rows)
             .sort_values("R2", ascending=False)
             .reset_index(drop=True))
    table.index = table.index + 1
    table.index.name = "Rank"
    return table, fitted


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------
def _scores_for_auc(model, X):
    """Probability or decision scores - never hard predictions."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1], "predict_proba"
    if hasattr(model, "decision_function"):
        return model.decision_function(X), "decision_function"
    return None, "unavailable"


def classification_metrics(y_true, y_pred, y_score=None,
                           pos_label=config.CLASSIFICATION["positive_class"]) -> dict:
    """Averaging is explicit:
      - Precision / Recall: BINARY, positive class = failure (Target == 1)
      - F1: WEIGHTED, as the rubric mandates
    """
    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision_failure": float(precision_score(y_true, y_pred, pos_label=pos_label,
                                                   zero_division=0)),
        "Recall_failure": float(recall_score(y_true, y_pred, pos_label=pos_label,
                                             zero_division=0)),
        "F1_weighted": float(f1_score(y_true, y_pred, average="weighted",
                                      zero_division=0)),
        "F1_failure": float(f1_score(y_true, y_pred, pos_label=pos_label,
                                     zero_division=0)),
    }
    out["ROC_AUC"] = float(roc_auc_score(y_true, y_score)) if y_score is not None else np.nan
    return out


def evaluate_classifiers(models: dict, X_train, y_train, X_test, y_test,
                         verbose: bool = True) -> tuple[pd.DataFrame, dict, dict]:
    rows, fitted, cms = [], {}, {}
    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        y_pred = model.predict(X_test)
        y_score, how = _scores_for_auc(model, X_test)
        m = classification_metrics(y_test, y_pred, y_score)
        m.update({"Model": name, "AUC_source": how, "Fit_seconds": round(fit_s, 3)})
        rows.append(m)
        fitted[name] = model
        cms[name] = confusion_matrix(y_test, y_pred)
        if verbose:
            print(f"  {name:<34} acc={m['Accuracy']:.4f} F1w={m['F1_weighted']:.4f} "
                  f"rec(fail)={m['Recall_failure']:.4f} AUC={m['ROC_AUC']:.4f}")
    cols = ["Model", "Accuracy", "Precision_failure", "Recall_failure",
            "F1_weighted", "F1_failure", "ROC_AUC", "AUC_source", "Fit_seconds"]
    table = (pd.DataFrame(rows)[cols]
             .sort_values("F1_weighted", ascending=False)
             .reset_index(drop=True))
    table.index = table.index + 1
    table.index.name = "Rank"
    return table, fitted, cms


def majority_class_baseline(y_train, y_test) -> dict:
    """Reference point that contextualises class imbalance."""
    major = pd.Series(y_train).value_counts().idxmax()
    y_pred = np.full(len(y_test), major)
    return {
        "majority_class": int(major),
        "baseline_accuracy": float(accuracy_score(y_test, y_pred)),
        "baseline_f1_weighted": float(f1_score(y_test, y_pred, average="weighted",
                                               zero_division=0)),
        "test_class_counts": pd.Series(y_test).value_counts().sort_index().to_dict(),
    }


# --------------------------------------------------------------------------
# Cross-validation (training data only)
# --------------------------------------------------------------------------
def cv_scores(models: dict, X_train, y_train, scoring: str,
              cv=config.CV_FOLDS) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        s = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring,
                            n_jobs=-1)
        rows.append({
            "Model": name, "scoring": scoring, "folds": cv,
            "CV_mean": float(np.mean(s)), "CV_std": float(np.std(s)),
            "fold_scores": [round(float(v), 5) for v in s],
        })
        print(f"  {name:<34} {scoring} CV = {np.mean(s):.4f} (+/- {np.std(s):.4f})")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def save_table(df: pd.DataFrame, name: str, also_markdown: bool = True) -> Path:
    config.ensure_dirs()
    csv = config.TABLES_DIR / f"{name}.csv"
    df.to_csv(csv)
    if also_markdown:
        (config.TABLES_DIR / f"{name}.md").write_text(df.to_markdown())
    print(f"[saved] {csv.relative_to(config.PROJECT_ROOT)}")
    return csv


def save_json(obj, name: str) -> Path:
    config.ensure_dirs()
    p = config.TABLES_DIR / f"{name}.json"
    p.write_text(json.dumps(obj, indent=2, default=str))
    print(f"[saved] {p.relative_to(config.PROJECT_ROOT)}")
    return p
