# Experiment Log — Review 1

Chronological record of what was actually run. Machine-readable equivalents:
`results/tables/run_log_*.json`.

Environment: Python 3.12, scikit-learn 1.8.0, pandas 3.0.2, numpy 2.4.4.
Seed `random_state=42` throughout.

---

## Data acquisition

| Step | Outcome |
|---|---|
| Searched workspace for CSVs | none present initially |
| Attempted Kaggle download | **blocked** — sandbox network denied `kaggle.com` (`host_not_allowed`) |
| Team supplied the CSVs manually | ✅ both verified |

Shape verification — **both matched the contract exactly**:
`Student_Performance.csv` 10,000×6 · `predictive_maintenance.csv` 10,000×10.
Checksums in `results/tables/dataset_provenance.json`.

Findings at audit: 0 missing cells in both; **127 exact duplicate rows** in the
regression data; classification target 9,661 / 339 (**3.39 % positive**).

---

## Scope reduction to Review 1

This repository was derived from the full three-track project by removing the
clustering track and classification Part B. What was verified during that split:

| Check | Result |
|---|---|
| Part A metrics before vs after removing Part B | **byte-identical** — Logistic 0.9560, KNN 0.9679, GNB 0.9506, DT 0.9714, SVC 0.9635 |
| Regression metrics before vs after | **byte-identical** |
| Reason they are identical | Part A models train on the same stratified split with the same seed; Part B never influenced them |

The split therefore changed **scope**, not results. Two things did legitimately
change:

1. **The CV leaders differ.** In the full project the two leading classifiers
   were Bagging and MLP (Part B). Within Part A the leaders are the Decision
   Tree and KNN, so those are what get tuned here.
2. **The app now serves a Part A model.** It previously loaded Bagging (Part B);
   it now loads the tuned Decision Tree, so the GUI demonstrates only Review 1
   work.

---

## Full runs (`--mode full`) — these are the reported figures

| Track | Key outcome |
|---|---|
| regression | best test R² 0.98899 (Polynomial deg 2); all 10 within 1.2 % R² |
| classification Part A | best weighted F1 0.9714 (Decision Tree); failure recall spans 0.10–0.38 untuned |

Total runtime ≈ 0.7 min for both tracks.

---

## Notable honest results

- **Regression tuning moved CV R² in the 5th decimal.** Ridge `alpha=0.1`,
  Lasso `alpha=0.001`. Recorded as-is.
- **Decision Tree tuning nearly doubled failure recall** (0.3824 → 0.7353) and
  raised CV weighted F1 (0.97480 → 0.98168) — **but ROC-AUC fell** (0.9219 →
  0.8847). A deeper tree decides better at the 0.5 threshold and ranks worse
  across thresholds. Both directions are reported.
- **KNN tuning barely moved anything**: recall 0.2941 → 0.3088, and ROC-AUC fell
  0.8291 → 0.7865.
- **Lasso zeroed 0 of 5 coefficients** — no sparsity to report.
- **Polynomial degree 3 was worse than degree 2** on test R² (0.98896 vs
  0.98899) despite 55 terms vs 20.
- **Gaussian Naive Bayes is last of the five** (F1w 0.9506), consistent with the
  two strongly correlated feature pairs (air/process temperature r = 0.876,
  speed/torque r = −0.875) violating its independence assumption.

---

## Verification

| Check | Result |
|---|---|
| `pytest tests/` | 12/12 passed |
| Notebook execution | both ran top-to-bottom, **0 cell errors** |
| Stored R² reproduces from scratch | exact to 1e-9 |
| Saved pipelines round-trip | identical predictions from raw-shaped input |
| Scope guard | clustering + Part B confirmed absent |

---

## Not done

- Public deployment (no URL claimed)
- All team-authored analysis (deliberately)
