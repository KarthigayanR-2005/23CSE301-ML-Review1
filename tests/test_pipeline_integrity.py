"""Focused tests for the risks that would actually invalidate the results.

Deliberately NOT a test-every-function suite. Each test here targets a failure
mode that would cost marks or make a number wrong:
  1. data leakage (a fitted scaler seeing the test set)
  2. the excluded-column contract (Failure Type is target leakage)
  3. model counts and the exact algorithms the PDF names for Review 1
  4. metric correctness (ROC-AUC from scores, target left in original units)
  5. saved pipelines predicting the same thing after a round trip

Run:  python -m pytest tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from src import config
from src import classification_models as clf_m
from src import data_loading as dl
from src import evaluation as ev
from src import preprocessing as pp
from src import regression_models as reg_m

RAW_PRESENT = all((config.RAW_DIR / s["filename"]).exists()
                  for s in config.DATASETS.values())
needs_data = pytest.mark.skipif(not RAW_PRESENT, reason="raw CSVs not in data/raw/")


# ---------------------------------------------------------------- leakage
def test_scaler_is_fitted_per_fold_not_on_full_data():
    """A scaler inside a Pipeline must see only the training rows of a fold.

    Constructed so that train and test have deliberately different means: if the
    scaler had been fitted on everything, the transformed training mean would
    not be ~0.
    """
    from sklearn.preprocessing import StandardScaler
    train = pd.DataFrame({"a": np.arange(100.0), "b": np.arange(100.0) * 2})
    test = pd.DataFrame({"a": np.arange(1000.0, 1100.0), "b": np.arange(100.0)})
    prep = pp.make_preprocessor(["a", "b"], [], scale=True)
    prep.fit(train)
    tr = prep.transform(train)
    assert abs(tr.mean()) < 1e-9, "training data should standardise to mean 0"
    te = prep.transform(test)
    assert te[:, 0].mean() > 10, "test set must be transformed with TRAIN statistics"


def test_every_model_is_a_pipeline_with_preprocessing():
    models = {**reg_m.build_models(["Hours Studied"], []),
              **clf_m.build_all(["Torque [Nm]"], [])}
    for name, m in models.items():
        assert isinstance(m, Pipeline), f"{name} is not a Pipeline"
        assert "prep" in m.named_steps, f"{name} has no preprocessing step"


# ------------------------------------------------------- column contracts
@needs_data
def test_failure_type_is_excluded_from_predictors():
    """Failure Type is a second outcome of the same event as Target."""
    prep = dl.prepare("classification", verbose=False)
    assert "Failure Type" not in prep.columns
    assert "UDI" not in prep.columns and "Product ID" not in prep.columns
    assert prep.shape[1] == 7


@needs_data
def test_raw_shapes_match_the_documented_contract():
    for track, spec in config.DATASETS.items():
        df = pd.read_csv(config.RAW_DIR / spec["filename"])
        assert tuple(df.shape) == tuple(spec["expected_raw_shape"]), track


# ------------------------------------------------------------ model counts
def test_algorithm_counts_and_identities():
    """Review 1 scope: 10 regressors, 5 Part A classifiers, no Part B."""
    r = reg_m.build_models(["Hours Studied"], [])
    assert len(r) == 10
    a = clf_m.build_part_a(["Torque [Nm]"], [])
    assert len(a) == 5
    # the PDF names Gaussian NB specifically
    assert isinstance(a["A3. Gaussian Naive Bayes"].named_steps["model"], GaussianNB)
    # Part B must NOT be present in this repository
    assert not hasattr(clf_m, "build_part_b"), "Part B belongs to Review 2"
    assert all(k.startswith("A") for k in a), "Part A keys only"


# ---------------------------------------------------------------- metrics
def test_roc_auc_uses_scores_not_hard_labels():
    """Hard labels would give a materially different (worse) AUC."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=400)})
    y = (X["x"] + rng.normal(scale=0.5, size=400) > 0).astype(int)
    m = LogisticRegression().fit(X, y)
    score, how = ev._scores_for_auc(m, X)
    assert how == "predict_proba"
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, score) > roc_auc_score(y, m.predict(X))


def test_regression_metrics_are_in_original_target_units():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    m = ev.regression_metrics(y, y + 2.0)
    assert m["MAE"] == pytest.approx(2.0)
    assert m["RMSE"] == pytest.approx(2.0)


def test_precision_recall_use_the_failure_class_as_positive():
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 1, 0])      # 1 of 2 failures caught
    m = ev.classification_metrics(y_true, y_pred)
    assert m["Recall_failure"] == pytest.approx(0.5)
    assert m["Precision_failure"] == pytest.approx(1.0)


def test_majority_baseline_reflects_imbalance():
    y_tr = pd.Series([0] * 95 + [1] * 5)
    y_te = pd.Series([0] * 19 + [1] * 1)
    b = ev.majority_class_baseline(y_tr, y_te)
    assert b["majority_class"] == 0
    assert b["baseline_accuracy"] == pytest.approx(0.95)


# ------------------------------------------------------- feature engineering
def test_feature_hook_is_a_no_op_until_the_team_registers_one():
    from src import feature_engineering as fe
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert fe.FeatureAdder(track="regression").transform(df).equals(df)
    assert fe.status("regression")["rubric_B3"] in ("INCOMPLETE", "COMPLETE")


# ------------------------------------------------------------ round-tripping
@needs_data
def test_saved_pipeline_round_trips_identically(tmp_path):
    import joblib
    df = dl.prepare("regression", verbose=False)
    Xtr, Xte, ytr, _ = pp.split_supervised(df, config.REGRESSION["target"])
    m = reg_m.build_models(config.REGRESSION["numeric_features"],
                           config.REGRESSION["categorical_features"])["1. Linear Regression"]
    m.fit(Xtr, ytr)
    before = m.predict(Xte.head(20))
    p = tmp_path / "m.joblib"
    joblib.dump(m, p)
    after = joblib.load(p).predict(Xte.head(20))
    np.testing.assert_allclose(before, after)


@needs_data
def test_split_is_reproducible_with_the_fixed_seed():
    df = dl.prepare("regression", verbose=False)
    a = pp.split_supervised(df, config.REGRESSION["target"])[0]
    b = pp.split_supervised(df, config.REGRESSION["target"])[0]
    pd.testing.assert_frame_equal(a, b)
