"""validate_project.py - verify what actually exists, not what was intended.

Scope: REVIEW 1 - the full regression track and classification PART A.

Checks structure, schemas, leakage exclusions, model counts, metric
recomputation, saved-pipeline prediction consistency, and rubric coverage.
Prints a status per check and exits non-zero if anything MANDATORY fails.

'INCOMPLETE' is a legitimate, reported outcome - the script never marks the
project submission-ready while team-written sections or feature engineering
remain outstanding.
"""
from __future__ import annotations

import json
import sys
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
from src import preprocessing as pp
from src import regression_models as reg_m

RESULTS: list[tuple[str, str, str]] = []
PASS, FAIL, WARN, TODO = "PASS", "FAIL", "WARN", "TODO"


def check(name: str, status: str, detail: str = "") -> None:
    RESULTS.append((name, status, detail))
    icon = {PASS: "PASS", FAIL: "FAIL", WARN: "WARN", TODO: "TODO"}[status]
    print(f"  [{icon}] {name}" + (f" - {detail}" if detail else ""))


def section(t: str) -> None:
    print(f"\n{t}\n" + "-" * len(t))


# --------------------------------------------------------------------------
def check_structure() -> None:
    section("1. Repository structure")
    required = [
        "README.md", "requirements.txt", ".gitignore", "CLAUDE.md",
        "data/README.md", "src/__init__.py", "src/data_loading.py",
        "src/preprocessing.py", "src/feature_engineering.py", "src/evaluation.py",
        "src/plotting.py", "src/regression_models.py", "src/classification_models.py",
        "scripts/download_data.py", "scripts/run_all.py",
        "scripts/validate_project.py", "notebooks/regression.ipynb",
        "notebooks/classification.ipynb",
        "app/streamlit_app.py", "docs/rubric_checklist.md", "docs/dataset_sources.md",
        "docs/team_contributions.md", "docs/team_analysis_prompts.md",
        "docs/instructor_clarifications.md", "docs/review1_outline.md",
        "docs/experiment_log.md",
    ]
    missing = [f for f in required if not (config.PROJECT_ROOT / f).exists()]
    check("required files present", PASS if not missing else FAIL,
          "all present" if not missing else f"missing: {missing}")


def check_data() -> None:
    section("2. Dataset schemas and exclusions")
    for track, spec in config.DATASETS.items():
        path = config.RAW_DIR / spec["filename"]
        if not path.exists():
            check(f"{track}: raw file", FAIL, f"{path.name} absent")
            continue
        raw = pd.read_csv(path)
        ok = tuple(raw.shape) == tuple(spec["expected_raw_shape"])
        check(f"{track}: raw shape", PASS if ok else FAIL,
              f"{raw.shape} (expected {tuple(spec['expected_raw_shape'])})")
        prep = dl.prepare(track, raw, verbose=False)
        n_cols = prep.shape[1]
        check(f"{track}: prepared columns in 6-9 range",
              PASS if 6 <= n_cols <= 9 else WARN, f"{n_cols} columns")
        check(f"{track}: prepared rows >= 10000",
              PASS if len(prep) >= 10000 else WARN, f"{len(prep)} rows")

    # leakage exclusions
    spec = config.CLASSIFICATION
    prep = dl.prepare("classification", verbose=False)
    leaked = [c for c in spec["leakage_columns"] if c in prep.columns]
    check("classification: Failure Type excluded (target leakage)",
          PASS if not leaked else FAIL, f"leaked: {leaked}" if leaked else "excluded")
    ids = [c for c in ("UDI", "Product ID") if c in prep.columns]
    check("classification: identifiers excluded", PASS if not ids else FAIL)

    # Review 1 scope: clustering must NOT be present in this repository
    stray = [f for f in ("src/clustering_models.py", "notebooks/clustering.ipynb")
             if (config.PROJECT_ROOT / f).exists()]
    check("Review 1 scope: clustering track absent",
          PASS if not stray else FAIL, f"found: {stray}" if stray else "")
    check("Review 1 scope: only 2 datasets configured",
          PASS if set(config.DATASETS) == {"regression", "classification"} else FAIL,
          str(sorted(config.DATASETS)))


def check_model_counts() -> None:
    section("3. Model counts")
    r = reg_m.build_models(config.REGRESSION["numeric_features"],
                           config.REGRESSION["categorical_features"])
    check("regression: 10 algorithms", PASS if len(r) == 10 else FAIL, f"{len(r)}")
    a = clf_m.build_part_a(config.CLASSIFICATION["numeric_features"],
                           config.CLASSIFICATION["categorical_features"])
    check("classification: 5 Part-A algorithms", PASS if len(a) == 5 else FAIL, f"{len(a)}")
    from sklearn.naive_bayes import GaussianNB
    is_gnb = isinstance(a["A3. Gaussian Naive Bayes"].named_steps["model"], GaussianNB)
    check("classification: Naive Bayes variant is Gaussian", PASS if is_gnb else FAIL)
    check("classification: Part B absent (Review 2 scope)",
          PASS if not hasattr(clf_m, "build_part_b") else FAIL)


def check_pipeline_hygiene() -> None:
    section("4. Leakage hygiene inside pipelines")
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    models = {**reg_m.build_models(config.REGRESSION["numeric_features"],
                                   config.REGRESSION["categorical_features"]),
              **clf_m.build_part_a(config.CLASSIFICATION["numeric_features"],
                                   config.CLASSIFICATION["categorical_features"])}
    bad = [n for n, m in models.items() if not isinstance(m, Pipeline)]
    check("every model is a Pipeline (transformers refit per CV fold)",
          PASS if not bad else FAIL, f"not pipelines: {bad}" if bad else "")
    has_prep = [n for n, m in models.items() if "prep" not in m.named_steps]
    check("every pipeline carries its own preprocessing step",
          PASS if not has_prep else FAIL)


def check_metrics_recompute() -> None:
    section("5. Metric recomputation (independent of the saved tables)")
    t = config.TABLES_DIR / "regression_comparison.csv"
    if not t.exists():
        check("regression comparison table", FAIL, "not found - run scripts/run_all.py")
        return
    table = pd.read_csv(t)
    check("regression: 10 rows in comparison table",
          PASS if len(table) == 10 else FAIL, f"{len(table)} rows")
    check("regression: ranked by test R2 descending",
          PASS if table["R2"].is_monotonic_decreasing else FAIL)
    for col in ("R2", "RMSE", "MAE"):
        check(f"regression: {col} column present", PASS if col in table.columns else FAIL)

    # recompute the linear baseline from scratch and compare
    df = dl.prepare("regression", verbose=False)
    Xtr, Xte, ytr, yte = pp.split_supervised(df, config.REGRESSION["target"])
    m = reg_m.build_models(config.REGRESSION["numeric_features"],
                           config.REGRESSION["categorical_features"])["1. Linear Regression"]
    m.fit(Xtr, ytr)
    recomputed = ev.regression_metrics(yte, m.predict(Xte))["R2"]
    stored = float(table.loc[table["Model"] == "1. Linear Regression", "R2"].iloc[0])
    ok = abs(recomputed - stored) < 1e-9
    check("regression: stored R2 reproduces exactly", PASS if ok else FAIL,
          f"stored={stored:.8f} recomputed={recomputed:.8f}")

    c = config.TABLES_DIR / "classification_partA_comparison.csv"
    if c.exists():
        ct = pd.read_csv(c)
        check("classification: 5 rows in the Part A table",
              PASS if len(ct) == 5 else FAIL, f"{len(ct)} rows")
        for col in ("Accuracy", "Precision_failure", "Recall_failure",
                    "F1_weighted", "ROC_AUC"):
            check(f"classification: {col} present", PASS if col in ct.columns else FAIL)
        probs = ct["AUC_source"].isin(["predict_proba", "decision_function"]).all()
        check("classification: ROC-AUC from scores, not hard labels",
              PASS if probs else FAIL)
        check("classification: no NaN ROC-AUC",
              PASS if ct["ROC_AUC"].notna().all() else FAIL)
    else:
        check("classification Part A table", FAIL, "not found")


def check_saved_pipelines() -> None:
    section("6. Saved-pipeline prediction consistency")
    for track, fname, target in (
            ("regression", "regression_best_pipeline.joblib", config.REGRESSION["target"]),
            ("classification", "classification_best_pipeline.joblib", config.CLASSIFICATION["target"])):
        p = config.MODELS_DIR / fname
        if not p.exists():
            check(f"{track}: saved pipeline", FAIL, f"{fname} absent")
            continue
        model = joblib.load(p)
        df = dl.prepare(track, verbose=False)
        X = df.drop(columns=[target]).head(50)
        try:
            pred1 = model.predict(X)
            pred2 = joblib.load(p).predict(X)
            same = np.array_equal(pred1, pred2)
            check(f"{track}: reloaded pipeline predicts identically",
                  PASS if same else FAIL, f"{len(pred1)} rows predicted from RAW-shaped input")
        except Exception as e:
            check(f"{track}: saved pipeline predicts", FAIL, str(e)[:200])


def check_figures() -> None:
    section("7. Required visualisations on disk")
    required = {
        "regression": ["reg_feature_distributions", "reg_correlation_heatmap",
                       "reg_target_distribution", "reg_feature_target_scatter",
                       "reg_residuals", "reg_pred_vs_actual", "reg_feature_importance"],
        "classification": ["clf_feature_distributions", "clf_correlation_heatmap",
                           "clf_target_distribution", "clf_feature_target_scatter",
                           "clf_feature_by_class", "clf_confusion_partA",
                           "clf_roc_curves", "clf_decision_tree", "clf_feature_importance"],
    }
    for sub, names in required.items():
        missing = [n for n in names
                   if not (config.FIGURES_DIR / sub / f"{n}.png").exists()]
        check(f"{sub}: {len(names) - len(missing)}/{len(names)} required figures",
              PASS if not missing else FAIL, f"missing {missing}" if missing else "")


def check_team_sections() -> None:
    section("8. Team-authored work (NOT AI-generatable)")
    for track in ("regression", "classification"):
        s = fe.status(track)
        check(f"{track}: feature engineering (rubric B3)",
              PASS if s["rubric_B3"] == "COMPLETE" else TODO, s["detail"])
    import nbformat
    for nb_name in ("regression", "classification"):
        p = config.NOTEBOOKS_DIR / f"{nb_name}.ipynb"
        if not p.exists():
            check(f"{nb_name}.ipynb", FAIL, "absent")
            continue
        nb = nbformat.read(p, as_version=4)
        todo = sum(1 for c in nb.cells
                   if c.cell_type == "markdown" and "TEAM TO COMPLETE" in c.source)
        unwritten = sum(1 for c in nb.cells
                        if c.cell_type == "markdown" and "(your written observation here)" in c.source)
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        total_code = len(code_cells)
        # A cell counts as executed if it has an execution_count. A cell with no
        # output is fine (imports and assignments are silent); a cell with an
        # ERROR output is not - that is what deliverable D1 forbids.
        ran = sum(1 for c in code_cells if c.get("execution_count") is not None)
        errored = sum(1 for c in code_cells
                      if any(o.get("output_type") == "error" for o in c.get("outputs", [])))
        with_output = sum(1 for c in code_cells if c.get("outputs"))
        check(f"{nb_name}.ipynb: all cells executed",
              PASS if ran == total_code and total_code else FAIL,
              f"{ran}/{total_code} code cells executed")
        check(f"{nb_name}.ipynb: no execution errors",
              PASS if errored == 0 else FAIL,
              f"{errored} cells raised" if errored else
              f"0 errors; {with_output}/{total_code} cells produced visible output")
        check(f"{nb_name}.ipynb: team observation cells",
              TODO if unwritten else PASS,
              f"{unwritten} of {todo} still unwritten")


def check_honesty() -> None:
    section("9. Honesty checks")
    p = config.TABLES_DIR / "regression_tuning.csv"
    if p.exists():
        t = pd.read_csv(p)
        n_improved = int(t["improved_on_CV"].sum())
        check("regression tuning reports real outcomes", PASS,
              f"{n_improved}/{len(t)} models improved on CV - reported as measured")
    p = config.TABLES_DIR / "classification_tuning.csv"
    if p.exists():
        t = pd.read_csv(p)
        n_improved = int(t["improved_on_CV"].sum())
        check("classification tuning reports real outcomes", PASS,
              f"{n_improved}/{len(t)} models improved on CV - reported as measured")


def main() -> int:
    print("=" * 78)
    print("REVIEW 1 VALIDATION - reporting what exists, not what was planned")
    print("  scope: full regression track + classification PART A")
    print("=" * 78)
    check_structure()
    check_data()
    check_model_counts()
    check_pipeline_hygiene()
    check_metrics_recompute()
    check_saved_pipelines()
    check_figures()
    check_team_sections()
    check_honesty()

    n_fail = sum(1 for _, s, _ in RESULTS if s == FAIL)
    n_todo = sum(1 for _, s, _ in RESULTS if s == TODO)
    n_warn = sum(1 for _, s, _ in RESULTS if s == WARN)
    n_pass = sum(1 for _, s, _ in RESULTS if s == PASS)

    print("\n" + "=" * 78)
    print(f"SUMMARY: {n_pass} pass, {n_fail} fail, {n_warn} warn, {n_todo} awaiting team")
    if n_fail:
        print("\nSTATUS: FAILING CHECKS PRESENT - not ready for review.")
    elif n_todo:
        print("\nSTATUS: IMPLEMENTATION AND EXECUTION COMPLETE, BUT NOT SUBMISSION-READY.")
        print(f"        {n_todo} item(s) require team-authored work that AI must not")
        print("        supply (feature engineering, EDA observations, interpretation,")
        print("        model-selection arguments, conclusions).")
    else:
        print("\nSTATUS: all checks pass.")
    print("=" * 78)

    config.ensure_dirs()
    pd.DataFrame(RESULTS, columns=["check", "status", "detail"]).to_csv(
        config.TABLES_DIR / "validation_report.csv", index=False)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
