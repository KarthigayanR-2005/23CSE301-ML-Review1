"""run_all.py - execute the full REVIEW 1 project end to end.

Scope: the full regression track + classification PART A only.
Clustering and classification Part B belong to Review 2 and are not in this
repository.

    python scripts/run_all.py --track all --mode full
    python scripts/run_all.py --track regression --mode dev

--mode dev   : reduced grids / k-ranges / repeats. DEVELOPMENT RUN.
               Every artifact it writes is stamped 'DEVELOPMENT RUN' and goes
               to results/_smoke-style filenames suffixed '_dev'.
--mode full  : the run whose numbers are reported in the README and reviews.

Nothing here invents a result. Tuning outcomes are reported as measured, even
when tuning makes a model worse.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from src import config
from src import classification_models as clf_m
from src import data_loading as dl
from src import evaluation as ev
from src import feature_engineering as fe
from src import plotting as pl
from src import preprocessing as pp
from src import regression_models as reg_m

LOG: list[dict] = []


def log(step: str, status: str, detail: str = "") -> None:
    entry = {"time": datetime.now().isoformat(timespec="seconds"),
             "step": step, "status": status, "detail": detail}
    LOG.append(entry)
    print(f"[{status}] {step} {detail}")


def suffix(mode: str) -> str:
    return "_dev" if mode == "dev" else ""


def banner(text: str) -> None:
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


# ==========================================================================
# REGRESSION
# ==========================================================================
def run_regression(mode: str = "full") -> dict:
    banner("REGRESSION TRACK - Student Performance (Review 1)")
    sfx, sub = suffix(mode), "regression"
    spec = config.REGRESSION
    out: dict = {"track": "regression", "mode": mode}

    # ---- load & audit -----------------------------------------------------
    raw = dl.load_raw("regression")
    dl.audit(raw, "regression", "raw")
    summary = dl.audit_summary(raw, "regression")
    out["raw_shape"] = summary["shape"]
    out["n_duplicate_rows"] = summary["n_duplicate_rows"]
    out["missing_cells"] = summary["total_missing_cells"]
    log("regression.audit", "EXECUTED", f"raw={summary['shape']} "
        f"dups={summary['n_duplicate_rows']} missing={summary['total_missing_cells']}")

    df = dl.prepare("regression", raw)
    out["prepared_shape"] = tuple(df.shape)

    # Duplicate policy: REPORTED, not auto-applied. See docs/team_analysis_prompts.md Q-RG1.
    df_dedup = pp.drop_exact_duplicates(df, verbose=False)
    out["rows_if_deduplicated"] = int(len(df_dedup))
    out["deduplication_conflicts_with_size_preference"] = bool(len(df_dedup) < 10000)
    ev.save_json({
        "prepared_rows": int(len(df)),
        "exact_duplicate_rows": int(len(df) - len(df_dedup)),
        "rows_after_dedup": int(len(df_dedup)),
        "size_preference_min_rows": 10000,
        "conflict": bool(len(df_dedup) < 10000),
        "policy": ("Duplicates are RETAINED for the reported run. All five inputs are "
                   "low-cardinality integers/binary, so identical rows are expected by "
                   "chance in a synthetic 10k sample and are not evidence of corrupted "
                   "records. TEAM TO CONFIRM - see docs/team_analysis_prompts.md Q-RG1."),
    }, f"regression_duplicate_policy{sfx}")

    # ---- EDA --------------------------------------------------------------
    feats = spec["numeric_features"] + spec["categorical_features"]
    pl.distribution_grid(df[feats], "Student Performance - input feature distributions",
                         f"reg_feature_distributions{sfx}", sub)
    pl.correlation_heatmap(df, "Student Performance - correlation heatmap (numeric)",
                           f"reg_correlation_heatmap{sfx}", sub)
    pl.target_distribution(df[spec["target"]],
                           "Target distribution - Performance Index",
                           f"reg_target_distribution{sfx}", sub)
    corr_t = df.select_dtypes("number").corr()[spec["target"]].drop(spec["target"])
    top2 = corr_t.abs().sort_values(ascending=False).index[:2].tolist()
    pl.feature_target_scatter(df, top2, spec["target"],
                              f"reg_feature_target_scatter{sfx}", sub)
    log("regression.eda", "EXECUTED", f"scatter features={top2}")

    # ---- split (BEFORE any data-driven decision) --------------------------
    X_train, X_test, y_train, y_test = pp.split_supervised(
        df, spec["target"], stratify=False, stratify_bins=None)
    out["split"] = {"train": list(X_train.shape), "test": list(X_test.shape),
                    "test_size": config.TEST_SIZE,
                    "stratified": False,
                    "note": "see docs/instructor_clarifications.md IC-2"}
    out["row_overlap"] = pp.check_no_row_leakage(X_train, X_test)
    log("regression.split", "EXECUTED", str(out["row_overlap"]))

    out["feature_engineering"] = fe.status("regression")

    # ---- all ten models on the held-out split -----------------------------
    models = reg_m.build_models(spec["numeric_features"], spec["categorical_features"])
    assert len(models) == 10, f"expected 10 regressors, built {len(models)}"
    print("\n-- held-out evaluation (10 models) --")
    table, fitted = ev.evaluate_regressors(models, X_train, y_train, X_test, y_test)
    ev.save_table(table, f"regression_comparison{sfx}")
    pl.model_comparison_bar(table, "R2", "Regression models ranked by test R2",
                            f"reg_model_comparison{sfx}", sub)
    out["comparison_table"] = table.to_dict("records")
    log("regression.models", "EXECUTED", f"{len(models)} models fitted")

    # ---- training-side CV selects the leading models ----------------------
    print("\n-- training cross-validation (model selection; test set untouched) --")
    cv_all = ev.cv_scores(models, X_train, y_train, scoring="r2")
    ev.save_table(cv_all.drop(columns="fold_scores"), f"regression_cv_all{sfx}")
    leaders = cv_all.sort_values("CV_mean", ascending=False)["Model"].head(2).tolist()
    out["cv_selected_leaders"] = leaders
    cv_top2 = cv_all[cv_all["Model"].isin(leaders)]
    ev.save_table(cv_top2.drop(columns="fold_scores"), f"regression_cv_top2{sfx}")
    ev.save_json(cv_top2.to_dict("records"), f"regression_cv_top2_folds{sfx}")
    log("regression.cv", "EXECUTED", f"5-fold CV R2; leaders={leaders}")

    # ---- hyperparameter tuning (>= 2 models) ------------------------------
    tunable = [m for m in leaders if m in reg_m.PARAM_GRIDS]
    for cand in reg_m.PARAM_GRIDS:
        if len(tunable) >= 2:
            break
        if cand not in tunable:
            tunable.append(cand)
    if mode == "dev":
        tunable = tunable[:2]

    print("\n-- hyperparameter tuning --")
    tuning_rows, tuned_models = [], {}
    for name in tunable:
        base_cv = float(cv_all.loc[cv_all["Model"] == name, "CV_mean"].iloc[0])
        base_test = float(table.loc[table["Model"] == name, "R2"].iloc[0])
        gs = reg_m.tune(name, models[name], X_train, y_train)
        tuned_test = ev.regression_metrics(y_test, gs.best_estimator_.predict(X_test))
        tuning_rows.append({
            "Model": name,
            "best_params": json.dumps(gs.best_params_),
            "CV_R2_before": round(base_cv, 5),
            "CV_R2_after": round(float(gs.best_score_), 5),
            "CV_delta": round(float(gs.best_score_) - base_cv, 5),
            "HeldOut_R2_before": round(base_test, 5),
            "HeldOut_R2_after": round(tuned_test["R2"], 5),
            "HeldOut_delta": round(tuned_test["R2"] - base_test, 5),
            "HeldOut_RMSE_after": round(tuned_test["RMSE"], 5),
            "improved_on_CV": bool(gs.best_score_ > base_cv),
        })
        tuned_models[name] = gs.best_estimator_
        pd.DataFrame(gs.cv_results_).to_csv(
            config.TUNING_DIR / f"regression_{name.split('.')[0]}_cv_results{sfx}.csv",
            index=False)
    tuning = pd.DataFrame(tuning_rows)
    ev.save_table(tuning, f"regression_tuning{sfx}")
    out["tuning"] = tuning.to_dict("records")
    log("regression.tuning", "EXECUTED",
        f"GridSearchCV on {len(tunable)} models; "
        f"{int(tuning['improved_on_CV'].sum())}/{len(tuning)} improved on CV")

    # ---- algorithm-specific displays --------------------------------------
    lin = fitted["1. Linear Regression"]
    coefs = reg_m.coefficient_table(lin)
    ev.save_table(coefs, f"regression_linear_coefficients{sfx}")
    out["linear_intercept"] = coefs.attrs["intercept"]

    spars = reg_m.sparsity_report(fitted["3. Lasso Regression"])
    ev.save_json(spars, f"regression_lasso_sparsity{sfx}")
    out["lasso_sparsity"] = spars

    degrees = (1, 2, 3) if mode == "full" else (1, 2)
    poly = reg_m.polynomial_degree_comparison(
        spec["numeric_features"], spec["categorical_features"],
        X_train, y_train, X_test, y_test, degrees=degrees)
    ev.save_table(poly, f"regression_polynomial_degrees{sfx}")
    out["polynomial_degrees"] = poly.to_dict("records")
    log("regression.algorithm_notes", "EXECUTED",
        f"coefficients, lasso sparsity ({spars['n_zero_coefficients']} zeroed), "
        f"poly degrees {degrees}")

    # ---- required visualisations ------------------------------------------
    best_name = table.iloc[0]["Model"] if table.index.name == "Rank" else table.loc[0, "Model"]
    best_name = table.sort_values("R2", ascending=False).iloc[0]["Model"]
    best_model = tuned_models.get(best_name, fitted[best_name])
    y_pred_best = best_model.predict(X_test)
    pl.residual_plot(y_test, y_pred_best, best_name, f"reg_residuals{sfx}", sub)
    pl.predicted_vs_actual(y_test, y_pred_best, best_name, f"reg_pred_vs_actual{sfx}", sub)
    imp = reg_m.tree_importances(fitted["7. Random Forest Regressor"])
    pl.importance_plot(imp.values, imp.index,
                       "Random Forest Regressor - feature importance",
                       f"reg_feature_importance{sfx}", sub, xlabel="Gini importance")
    pl.importance_plot(coefs["coefficient"].values, coefs["feature"],
                       "Linear Regression - coefficients (standardised inputs)",
                       f"reg_linear_coefficients{sfx}", sub, xlabel="Coefficient")
    out["best_model_heldout"] = best_name
    out["tree_importances"] = imp.round(5).to_dict()
    log("regression.figures", "EXECUTED", f"best={best_name}")

    # ---- persist ----------------------------------------------------------
    if mode == "full":
        config.ensure_dirs()
        joblib.dump(best_model, config.MODELS_DIR / "regression_best_pipeline.joblib")
        joblib.dump({"model_name": best_name, "features": list(X_train.columns),
                     "target": spec["target"], "random_state": config.RANDOM_STATE,
                     "sklearn_metrics": ev.regression_metrics(y_test, y_pred_best)},
                    config.MODELS_DIR / "regression_best_metadata.joblib")
        log("regression.persist", "EXECUTED", "saved fitted pipeline (preprocessing included)")

    ev.save_json(out, f"regression_run_summary{sfx}")
    return out


# ==========================================================================
# CLASSIFICATION - PART A ONLY (Review 1)
# ==========================================================================
def run_classification(mode: str = "full") -> dict:
    banner("CLASSIFICATION TRACK - Predictive Maintenance (PART A, Review 1)")
    sfx, sub = suffix(mode), "classification"
    spec = config.CLASSIFICATION
    cls_names = [spec["negative_class_name"], spec["positive_class_name"]]
    out: dict = {"track": "classification", "mode": mode, "part": "A (Review 1)"}

    raw = dl.load_raw("classification")
    dl.audit(raw, "classification", "raw")
    summary = dl.audit_summary(raw, "classification")
    out["raw_shape"] = summary["shape"]
    out["target_distribution_raw"] = summary["target_distribution"]
    out["leakage_columns_excluded"] = spec["leakage_columns"]
    log("classification.audit", "EXECUTED",
        f"raw={summary['shape']} target={summary['target_distribution']}")

    df = dl.prepare("classification", raw)
    out["prepared_shape"] = tuple(df.shape)
    assert spec["target"] in df.columns and "Failure Type" not in df.columns

    # ---- EDA --------------------------------------------------------------
    feats = spec["numeric_features"] + spec["categorical_features"]
    pl.distribution_grid(df[feats], "Predictive Maintenance - input feature distributions",
                         f"clf_feature_distributions{sfx}", sub)
    pl.correlation_heatmap(df, "Predictive Maintenance - correlation heatmap (numeric)",
                           f"clf_correlation_heatmap{sfx}", sub)
    pl.target_distribution(df[spec["target"]], "Target distribution - machine failure",
                           f"clf_target_distribution{sfx}", sub, discrete=True,
                           class_names={0: cls_names[0], 1: cls_names[1]})
    # Binary target -> jittered scatter plus a companion box plot
    pl.feature_target_scatter(df, ["Torque [Nm]", "Rotational speed [rpm]"],
                              spec["target"], f"clf_feature_target_scatter{sfx}", sub,
                              jitter=0.08,
                              title="Feature-target relationships (target jittered for readability)")
    pl.grouped_box(df, ["Torque [Nm]", "Rotational speed [rpm]", "Tool wear [min]"],
                   spec["target"], f"clf_feature_by_class{sfx}", sub,
                   class_names={0: cls_names[0], 1: cls_names[1]})
    log("classification.eda", "EXECUTED", "jittered scatter + companion box plots")

    # ---- split ------------------------------------------------------------
    X_train, X_test, y_train, y_test = pp.split_supervised(
        df, spec["target"], stratify=True)
    out["split"] = {"train": list(X_train.shape), "test": list(X_test.shape),
                    "stratified": True}
    out["row_overlap"] = pp.check_no_row_leakage(X_train, X_test)
    out["feature_engineering"] = fe.status("classification")

    baseline = ev.majority_class_baseline(y_train, y_test)
    ev.save_json(baseline, f"classification_majority_baseline{sfx}")
    out["majority_baseline"] = baseline
    log("classification.baseline", "EXECUTED",
        f"majority-class accuracy={baseline['baseline_accuracy']:.4f}")

    # ---- the five Part A models ------------------------------------------
    models = clf_m.build_part_a(spec["numeric_features"], spec["categorical_features"])
    assert len(models) == 5, f"expected 5 Part A classifiers, built {len(models)}"

    print("\n-- Part A (Review 1) held-out evaluation --")
    table, fitted, cms = ev.evaluate_classifiers(models, X_train, y_train, X_test, y_test)
    ev.save_table(table, f"classification_partA_comparison{sfx}")
    pl.confusion_grid(cms, cls_names, "Part A confusion matrices (held-out test set)",
                      f"clf_confusion_partA{sfx}", sub)
    pl.model_comparison_bar(table, "F1_weighted",
                            "Part A classifiers ranked by weighted F1",
                            f"clf_model_comparison{sfx}", sub)
    out["partA_table"] = table.to_dict("records")
    log("classification.partA", "EXECUTED", "5 models")

    pl.roc_curves(fitted, X_test, y_test,
                  "ROC curves (probability / decision scores, not hard labels)",
                  f"clf_roc_curves{sfx}", sub)

    # ---- CV on training data, leaders selected there ----------------------
    print("\n-- training cross-validation (model selection) --")
    cv_all = ev.cv_scores(models, X_train, y_train, scoring="f1_weighted")
    ev.save_table(cv_all.drop(columns="fold_scores"), f"classification_cv_all{sfx}")
    leaders = cv_all.sort_values("CV_mean", ascending=False)["Model"].head(2).tolist()
    cv_top2 = cv_all[cv_all["Model"].isin(leaders)]
    ev.save_table(cv_top2.drop(columns="fold_scores"), f"classification_cv_top2{sfx}")
    out["cv_selected_leaders"] = leaders
    log("classification.cv", "EXECUTED", f"leaders={leaders}")

    # ---- tuning -----------------------------------------------------------
    tunable = [m for m in leaders if m in clf_m.PARAM_GRIDS]
    for cand in clf_m.PARAM_GRIDS:
        if len(tunable) >= 2:
            break
        if cand not in tunable:
            tunable.append(cand)
    if mode == "dev":
        tunable = tunable[:2]

    print("\n-- hyperparameter tuning --")
    rows, tuned_models = [], {}
    for name in tunable:
        base_cv = float(cv_all.loc[cv_all["Model"] == name, "CV_mean"].iloc[0])
        base_row = table[table["Model"] == name].iloc[0]
        gs = clf_m.tune(name, models[name], X_train, y_train)
        y_pred = gs.best_estimator_.predict(X_test)
        score, _ = ev._scores_for_auc(gs.best_estimator_, X_test)
        after = ev.classification_metrics(y_test, y_pred, score)
        rows.append({
            "Model": name, "best_params": json.dumps(gs.best_params_),
            "CV_F1w_before": round(base_cv, 5),
            "CV_F1w_after": round(float(gs.best_score_), 5),
            "CV_delta": round(float(gs.best_score_) - base_cv, 5),
            "HeldOut_F1w_before": round(float(base_row["F1_weighted"]), 5),
            "HeldOut_F1w_after": round(after["F1_weighted"], 5),
            "HeldOut_Recall_failure_before": round(float(base_row["Recall_failure"]), 5),
            "HeldOut_Recall_failure_after": round(after["Recall_failure"], 5),
            "HeldOut_ROC_AUC_before": round(float(base_row["ROC_AUC"]), 5),
            "HeldOut_ROC_AUC_after": round(after["ROC_AUC"], 5),
            "improved_on_CV": bool(gs.best_score_ > base_cv),
        })
        tuned_models[name] = gs.best_estimator_
        pd.DataFrame(gs.cv_results_).to_csv(
            config.TUNING_DIR / f"classification_{name.split('.')[0]}_cv_results{sfx}.csv",
            index=False)
    tuning = pd.DataFrame(rows)
    ev.save_table(tuning, f"classification_tuning{sfx}")
    out["tuning"] = tuning.to_dict("records")
    log("classification.tuning", "EXECUTED",
        f"{int(tuning['improved_on_CV'].sum())}/{len(tuning)} improved on CV")

    # ---- algorithm-specific displays --------------------------------------
    odds = clf_m.odds_table(fitted["A1. Logistic Regression"])
    ev.save_table(odds, f"classification_logreg_odds{sfx}")
    pl.importance_plot(odds["coefficient_log_odds"].values, odds["feature"],
                       "Logistic Regression - coefficients (log-odds, per 1 SD)",
                       f"clf_logreg_coefficients{sfx}", sub, xlabel="Log-odds coefficient")

    dt = fitted["A4. Decision Tree Classifier"]
    pl.decision_tree_figure(dt.named_steps["model"],
                            dt.named_steps["prep"].get_feature_names_out(),
                            cls_names, f"clf_decision_tree{sfx}", sub, max_depth=3)
    imp = clf_m.tree_importances(dt)
    pl.importance_plot(imp.values, imp.index,
                       "Decision Tree Classifier - feature importance",
                       f"clf_feature_importance{sfx}", sub, xlabel="Gini importance")
    out["tree_importances"] = imp.round(5).to_dict()

    knn_cmp = clf_m.knn_distance_metric_comparison(
        spec["numeric_features"], spec["categorical_features"],
        X_train, y_train, X_test, y_test)
    ev.save_table(knn_cmp, f"classification_knn_distance_metrics{sfx}")
    out["knn_distance_metrics"] = knn_cmp.to_dict("records")
    log("classification.algorithm_notes", "EXECUTED",
        "odds ratios, tree visualisation, tree importance, KNN distance metrics")

    best_name = table.sort_values("F1_weighted", ascending=False).iloc[0]["Model"]
    out["best_model_heldout"] = best_name
    if mode == "full":
        best_model = tuned_models.get(best_name, fitted[best_name])
        joblib.dump(best_model, config.MODELS_DIR / "classification_best_pipeline.joblib")
        joblib.dump({"model_name": best_name, "features": list(X_train.columns),
                     "target": spec["target"],
                     "classes": {0: cls_names[0], 1: cls_names[1]},
                     "part": "A (Review 1)",
                     "random_state": config.RANDOM_STATE},
                    config.MODELS_DIR / "classification_best_metadata.joblib")
        log("classification.persist", "EXECUTED", f"saved {best_name}")

    ev.save_json(out, f"classification_run_summary{sfx}")
    return out


# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="all",
                    choices=["all", "regression", "classification"])
    ap.add_argument("--mode", default="full", choices=["full", "dev"])
    args = ap.parse_args()

    config.ensure_dirs()
    t0 = time.perf_counter()
    if args.mode == "dev":
        print("\n*** DEVELOPMENT RUN - reduced grids. Results are NOT the reported "
              "figures and are written with a '_dev' suffix. ***\n")

    results = {}
    if args.track in ("all", "regression"):
        results["regression"] = run_regression(args.mode)
    if args.track in ("all", "classification"):
        results["classification"] = run_classification(args.mode)

    elapsed = time.perf_counter() - t0
    ev.save_json({"mode": args.mode, "track": args.track,
                  "elapsed_seconds": round(elapsed, 1),
                  "finished": datetime.now().isoformat(timespec="seconds"),
                  "log": LOG},
                 f"run_log_{args.track}{suffix(args.mode)}")
    banner(f"DONE in {elapsed/60:.1f} min  (mode={args.mode}, track={args.track})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
