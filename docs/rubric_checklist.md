# Rubric Checklist — Review 1 · Requirement → Artifact → Status

**Legend:** ✅ done & executed · ⚠️ done, team confirmation needed · ❌ requires
team-authored work (AI must not supply it) · ➖ not in scope

Live version: `python scripts/validate_project.py` → `results/tables/validation_report.csv`.

> **Scope.** This repository covers **Review 1 only**: the full regression track
> and classification **Part A**. Review 2 (classification Part B + clustering)
> is a separate repository with its own checklist.

---

# Review 1 — 25 marks

## Section A — Dataset & EDA (4)

| # | Criterion | Max | Where | Status |
|---|---|---|---|---|
| A1 | Shape, dtypes, missing counts, target distribution | 1 | `regression.ipynb` §1 · `classification.ipynb` §1 · `results/tables/*_audit_raw.csv` | ✅ |
| A2 | Distribution plot per feature | 2 | `reg_feature_distributions.png`, `clf_feature_distributions.png` | ✅ |
| A2 | Correlation heatmap | ↑ | `reg_correlation_heatmap.png`, `clf_correlation_heatmap.png` | ✅ |
| A2 | Target distribution | ↑ | `reg_target_distribution.png`, `clf_target_distribution.png` | ✅ |
| A2 | ≥2 feature–target scatter plots | ↑ | `reg_feature_target_scatter.png` (2 panels) · `clf_feature_target_scatter.png` (2, jittered) + `clf_feature_by_class.png` companion | ✅ |
| A3 | **Written observation after each major visualisation** | 1 | ✍️ cells in both notebooks | ❌ **team** |

## Section B — Preprocessing & Feature Engineering (3)

| # | Criterion | Max | Where | Status |
|---|---|---|---|---|
| B1 | Missing values, justified strategy | 1 | `preprocessing.make_preprocessor` (median / most-frequent, inside pipelines). **0 missing cells in both datasets** | ✅ |
| B1 | Duplicates checked and treated | ↑ | 127 found in regression; `results/tables/regression_duplicate_policy.json` | ⚠️ **Q-RG1** |
| B1 | Outliers checked, not blanket-deleted | ↑ | distribution grids + `grouped_box`; nothing auto-deleted | ⚠️ team to comment |
| B2 | Encoding for categoricals | 1 | `OneHotEncoder(handle_unknown='ignore', drop='if_binary')` in every pipeline | ✅ |
| B2 | Scaler fitted on train only | ↑ | `StandardScaler` inside Pipeline → refit per CV fold. Test: `test_scaler_is_fitted_per_fold_not_on_full_data` | ✅ |
| B2 | Stratified split | ↑ | classification ✅ stratified; regression **not** stratified (continuous target) | ⚠️ **IC-2** |
| B3 | **≥1 engineered feature + written justification** | 1 | `src/feature_engineering.py` — **EMPTY BY DESIGN** | ❌ **team — Q-RG3 / Q-CF1** |

## Section C — Regression Track (9)

| # | Criterion | Max | Where | Status |
|---|---|---|---|---|
| C1 | All 10 algorithms trained, no errors | 4 | `regression_models.build_models` → 10 pipelines; `regression.ipynb` §4.1–4.10 | ✅ |
| C2 | Single table: R², RMSE, MAE, ranked by R² | 2 | `results/tables/regression_comparison.csv` — 10 rows, monotonic descending | ✅ |
| C3 | GridSearchCV on ≥2 models, best params + improvement | 2 | `results/tables/regression_tuning.csv` (Ridge, Lasso) + `results/tuning/*.csv` | ✅ *(gains are 5th-decimal — reported honestly)* |
| C4 | Residual plot + predicted-vs-actual for best model | 1 | `reg_residuals.png`, `reg_pred_vs_actual.png` | ✅ |
| C4 | Feature-importance plot, ≥1 tree model | ↑ | `reg_feature_importance.png` (Random Forest) | ✅ |
| — | 5-fold CV R² for the two best models | mandatory | `regression_cv_top2.csv` — leaders picked on **training** CV | ✅ |
| — | Coefficients (#1) | PDF note | `regression_linear_coefficients.csv` + plot | ✅ |
| — | Lasso sparsity (#3) | PDF note | `regression_lasso_sparsity.json` — **0 of 5 zeroed** | ✅ |
| — | Polynomial degree comparison (#5) | PDF note | `regression_polynomial_degrees.csv` — degrees 1, 2, 3 | ✅ |

## Section D — Classification Part A (3)

| # | Criterion | Max | Where | Status |
|---|---|---|---|---|
| D1 | All 5 Part-A algorithms, no errors | 2 | `build_part_a` — LogReg, KNN, **Gaussian**NB, DT, SVC; `classification.ipynb` §4.1–4.5 | ✅ |
| D2 | Accuracy, weighted F1, confusion matrix per algorithm | 1 | `classification_partA_comparison.csv`, `clf_confusion_partA.png` | ✅ |
| D2 | Preliminary comparison table | ↑ | same, plus `clf_model_comparison.png` | ✅ |
| D2 | Best model identified with justification | ↑ | table ranks Decision Tree first | ❌ **justification: team — Q-CF2** |
| — | Majority-class reference | requested | `classification_majority_baseline.json` — acc 0.9660, F1w 0.9493 | ✅ |
| — | ROC-AUC from probabilities | §7 | all five use `predict_proba`; `clf_roc_curves.png` | ✅ |
| — | CV for the two leading models | §7.2 | `classification_cv_top2.csv` | ✅ |
| — | Tuning with before/after | §7.2 | `classification_tuning.csv` — DT recall 0.38 → 0.74, AUC 0.92 → 0.88 | ✅ |
| — | Odds/coefficients (#1) | PDF note | `classification_logreg_odds.csv` + plot | ✅ |
| — | Distance metrics (#2) | PDF note | `classification_knn_distance_metrics.csv` | ✅ |
| — | Tree visualisation (#4) | PDF note | `clf_decision_tree.png` (depth-3 display) | ✅ |
| — | Feature importance | PDF note | `clf_feature_importance.png` (Decision Tree) | ✅ |

## Section E — Presentation (1) + Viva (5)

| # | Criterion | Max | Where | Status |
|---|---|---|---|---|
| E1 | Clear narrative; every member can explain their code | 1 | `docs/review1_outline.md` | ❌ **team** |
| — | Viva | 5 | `docs/team_analysis_prompts.md` ⚪ items | ❌ **team** |

---

## Deliverables (Review 1)

| # | Deliverable | Status |
|---|---|---|
| D1 | Notebooks, fully run, outputs visible | ✅ both executed, 0 errors |
| D2 | GitHub repo, meaningful commits | ✅ local history; push when ready |
| D3 | Comparative results table | ✅ regression (10 rows) + Part A (5 rows) |
| D4 | All required visualisations | ✅ 7/7 regression, 9/9 classification |

---

## Not in scope here (Review 2 repository)

➖ Classification Part B · ➖ consolidated 10-algorithm table · ➖ clustering
track, elbow, dendrogram, PCA/t-SNE · ➖ bonus deployment.

---

## Summary

**Every mechanical Review 1 requirement is implemented and executed.** What
remains is the marks that depend on your own words: **A3** (1), **B3** (1),
**D2 justification**, **E1** (1) and **viva** (5) — 8 of the 25 marks.
