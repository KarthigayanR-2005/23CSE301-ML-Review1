# 23CSE301 Machine Learning — Capstone Project · **REVIEW 1**

**B.Tech. Computer Science and Engineering, III Year · Academic Year 2026–27**
Three-member team

---

## 📌 Scope of this repository

This repository is the **Review 1 deliverable only**:

| Included | Excluded (Review 2) |
|---|---|
| ✅ Full regression track — all 10 algorithms | ❌ Classification **Part B** (RF, AdaBoost, Gradient Boosting, Bagging, MLP) |
| ✅ Classification **Part A** — the 5 Review 1 algorithms | ❌ Clustering track (K-Means, Agglomerative) |

The Review 2 work lives in a separate repository. Nothing here depends on it,
and `scripts/validate_project.py` actively checks that the clustering track and
Part B are absent, so the scope cannot drift by accident.

---

## ⚠️ Project status — read this first

| | |
|---|---|
| **Implemented** | 15 algorithms (10 regression, 5 classification Part A), full pipelines, all required visualisations, tuning, validation tooling, Streamlit GUI |
| **Executed** | Both tracks run end-to-end on the real data; both notebooks run top-to-bottom with outputs visible; 12/12 tests pass |
| **Awaiting team input** | Feature engineering (rubric B3) · every EDA observation · every interpretation · model-selection arguments · conclusions |

> **This project is NOT submission-ready.** Guidelines §7.5 requires that
> analysis, interpretation and feature-engineering decisions are the team's own
> work. Those sections are deliberately empty. Run
> `python scripts/validate_project.py` for the live checklist, and work through
> `docs/team_analysis_prompts.md`.

---

## 1. Overview

An end-to-end ML pipeline across two problem tracks: data loading, EDA,
cleaning, model training and comparison, hyperparameter tuning, and result
visualisation.

### Problem statements — **DRAFTS for team review** *(rewrite these yourselves)*

**Regression.** Given a student's study hours, previous scores, sleep hours,
practice-paper count and extracurricular participation, predict their
Performance Index. The practical question is which controllable habits carry
predictive weight once prior attainment is accounted for.

**Classification.** Given a machine's product-quality class, air and process
temperature, rotational speed, torque and tool wear, predict whether the machine
will fail. The operational question is whether failures can be caught early
enough to act on, given that failures are rare.

*(These are drafts written as scaffolding. Q-X1 in `docs/team_analysis_prompts.md`
asks you to replace them with your own wording.)*

---

## 2. Datasets

| Track | Dataset | Raw shape | Prepared shape | Target | Synthetic? |
|---|---|---|---|---|---|
| Regression | [Student Performance](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression) | 10,000 × 6 | 10,000 × 6 | `Performance Index` (continuous) | **Yes** |
| Classification | [Machine Predictive Maintenance](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) | 10,000 × 10 | 10,000 × 7 | `Target` (binary, 3.39 % positive) | **Yes** |

### 🔬 Synthetic-data disclosure
**Both datasets are synthetic.** No real student and no real machine was
measured. Nothing here supports a claim about real students or real equipment;
the datasets are teaching instruments with clean, well-behaved generating
processes — which is itself why nearly every regression model lands within 1 %
R² of every other.

### Exclusions and why

| Dataset | Excluded | Reason |
|---|---|---|
| Predictive Maintenance | `Failure Type` | **Target leakage** — a second recording of the same event as `Target` |
| Predictive Maintenance | `UDI`, `Product ID` | row identifiers |

Full provenance, checksums and licence pointers: **`docs/dataset_sources.md`**.

---

## 3. Setup

Python 3.11+ (tested on 3.12 and 3.13). Roughly 1 minute for the full run.

**macOS / Linux**
```bash
git clone <your-repo-url> && cd 23CSE301-ML-Review1
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py      # verifies data/raw/, prints instructions if absent
```

**Windows (PowerShell)**
```powershell
git clone <your-repo-url>; cd 23CSE301-ML-Review1
py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\download_data.py
```

The two CSVs must sit in `data/raw/`. They are **not** committed — `.gitignore`
excludes them. `scripts/download_data.py` tries the Kaggle API first and prints
exact manual steps otherwise. **Never put credentials in source code**; Kaggle
tokens belong in `~/.kaggle/kaggle.json`, which is git-ignored.

### Running everything

```bash
python scripts/run_all.py --mode full          # both tracks
python scripts/run_all.py --track regression   # one track
python scripts/run_all.py --mode dev           # fast development pass ('_dev' outputs)
python scripts/build_notebooks.py              # regenerate notebooks
python scripts/execute_notebooks.py            # run them top-to-bottom
python -m pytest tests/ -v                     # 12 focused tests
python scripts/validate_project.py             # rubric + integrity report
streamlit run app/streamlit_app.py             # GUI
```

`--mode dev` exists so a quick pass is never mistaken for the reported figures:
its outputs carry a `_dev` suffix and the run prints a `DEVELOPMENT RUN` banner.
**Every number below comes from `--mode full`.**

---

## 4. Methodology — how leakage is prevented

* Held-out 80:20 split established **before** any data-driven modelling decision.
* Every model is a `Pipeline(preprocess → estimator)`, so imputers, encoders and
  scalers are refitted on the training part of **every** CV fold — never on the
  full dataset (§7.1: fitting on full data is a mark deduction).
* Model selection uses **training-side** cross-validation. The held-out set is
  scored once, at the end, and never tuned against.
* The regression target is never scaled; R², RMSE and MAE are in Performance-Index points.
* `random_state=42` throughout.
* Classification uses a stratified split (3.39 % positive class). Regression does
  not — see `docs/instructor_clarifications.md` **IC-2**.

---

## 5. Results

### 5.1 Regression — all ten models, same held-out split, ranked by test R²

| Rank | Model | R² | RMSE | MAE | Train R² |
|---:|---|---:|---:|---:|---:|
| 1 | Polynomial (deg 2) + Linear | 0.988989 | 2.0201 | 1.6115 | 0.988705 |
| 2 | Linear Regression | 0.988983 | 2.0206 | 1.6111 | 0.988690 |
| 3 | Ridge Regression | 0.988982 | 2.0207 | 1.6112 | 0.988690 |
| 4 | Lasso Regression | 0.988969 | 2.0218 | 1.6117 | 0.988688 |
| 5 | ElasticNet Regression | 0.988886 | 2.0295 | 1.6174 | 0.988661 |
| 6 | Gradient Boosting Regressor | 0.988411 | 2.0723 | 1.6432 | 0.989104 |
| 7 | Random Forest Regressor | 0.986175 | 2.2635 | 1.8097 | 0.997498 |
| 8 | Support Vector Regressor | 0.985924 | 2.2839 | 1.7928 | 0.986171 |
| 9 | Decision Tree Regressor | 0.983901 | 2.4426 | 1.9514 | 0.985967 |
| 10 | K-Nearest Neighbors Regressor | 0.976898 | 2.9259 | 2.3603 | 0.984025 |

**5-fold CV R² (leaders nominated on training CV):** Linear Regression
0.98866 ± 0.00028 · Ridge Regression 0.98866 ± 0.00028.

**Tuning (GridSearchCV, 2 models).** Both improved — but in the **fifth decimal
place**, which is the honest result:

| Model | Best params | CV R² before → after | Held-out R² before → after |
|---|---|---|---|
| Ridge | `alpha=0.1` | 0.988660 → 0.988660 | 0.98898 → 0.98898 |
| Lasso | `alpha=0.001` | 0.988660 → 0.988660 | 0.98897 → 0.98898 |

**Polynomial degrees:** deg 1 → test R² 0.98898 · deg 2 → 0.98899 (20 terms) ·
deg 3 → 0.98896 (55 terms). **Lasso sparsity:** 0 of 5 coefficients driven to zero.

### 5.2 Classification Part A — the five Review 1 algorithms

Precision/Recall are **binary with failure (`Target == 1`) as the positive
class**; F1 is **weighted**; ROC-AUC uses `predict_proba` for all five models.
**Majority-class reference: accuracy 0.9660, weighted F1 0.9493.**

| Rank | Model | Accuracy | Precision (fail) | Recall (fail) | F1 (weighted) | ROC-AUC |
|---:|---|---:|---:|---:|---:|---:|
| 1 | A4. Decision Tree Classifier | 0.9755 | 0.7879 | 0.3824 | 0.9714 | 0.9219 |
| 2 | A2. K-Nearest Neighbors | 0.9740 | 0.8333 | 0.2941 | 0.9679 | 0.8291 |
| 3 | A5. Support Vector Classifier | 0.9720 | 0.8750 | 0.2059 | 0.9635 | 0.9468 |
| 4 | A1. Logistic Regression | 0.9675 | 0.6364 | 0.1029 | 0.9560 | 0.8994 |
| 5 | A3. Gaussian Naive Bayes | 0.9580 | 0.2500 | 0.1176 | 0.9506 | 0.8468 |

**5-fold CV weighted F1 (training data only):** Decision Tree 0.97480 ± 0.00174 ·
KNN 0.96786 ± 0.00243 · SVC 0.96677 ± 0.00177 · Logistic 0.96176 ± 0.00348 ·
Gaussian NB 0.95530 ± 0.00337.

**Tuning (GridSearchCV on the two CV leaders):**

| Model | Best params | CV F1w before → after | Held-out recall (fail) | Held-out ROC-AUC |
|---|---|---|---|---|
| A4. Decision Tree | `max_depth=None, min_samples_leaf=5, class_weight=None` | 0.97480 → **0.98168** | 0.3824 → **0.7353** | 0.9219 → **0.8847** |
| A2. KNN | `n_neighbors=3, weights=uniform` | 0.96786 → 0.96912 | 0.2941 → 0.3088 | 0.8291 → 0.7865 |

> **Read the Decision Tree row carefully — it is the most interesting result in
> this track.** Tuning nearly **doubled failure recall** (0.38 → 0.74), which is
> the metric that actually matters here. But ROC-AUC **fell** (0.92 → 0.88). A
> deeper tree with `min_samples_leaf=5` makes better hard decisions at the 0.5
> threshold while ranking cases *worse* across all thresholds. That trade-off is
> reported as measured, not smoothed over, and it is a genuine talking point for
> the viva.

**Decision Tree feature importance:** Torque 0.515 · Air temperature 0.147 ·
Tool wear 0.123 · Rotational speed 0.111 · Process temperature 0.100 ·
Type 0.005.

**KNN distance metrics:** euclidean F1w 0.9679 · manhattan 0.9671 ·
chebyshev 0.9628.

---

## 6. Repository structure

```
├── README.md                  ← you are here
├── requirements.txt           pinned to the versions actually tested
├── CLAUDE.md                  project rules + progress for later sessions
├── .gitignore                 credentials, venvs, caches, raw data
├── data/
│   ├── README.md              how to obtain the data
│   ├── raw/                   immutable source CSVs (git-ignored)
│   └── processed/             derived artifacts (regenerable)
├── notebooks/
│   ├── regression.ipynb       full regression track
│   └── classification.ipynb   classification Part A
├── src/                       the single shared implementation
│   ├── config.py              paths, seed, dataset contracts
│   ├── data_loading.py        loading, audit, provenance, checksums
│   ├── preprocessing.py       splits, column transformers, leakage checks
│   ├── feature_engineering.py ⚠️ EMPTY HOOK — team input required
│   ├── evaluation.py          metrics + comparison tables
│   ├── plotting.py            every figure
│   ├── regression_models.py   the 10 regressors + tuning grids
│   └── classification_models.py  the 5 Part A classifiers + tuning grids
├── scripts/
│   ├── download_data.py       acquisition + verification
│   ├── run_all.py             the execution engine
│   ├── algorithm_content.py   per-algorithm write-ups (one source of truth)
│   ├── build_notebooks.py     generates the notebooks
│   ├── execute_notebooks.py   runs them top-to-bottom
│   └── validate_project.py    rubric + integrity checks
├── results/{tables,figures,tuning}/   every computed artifact
├── models/                    saved pipelines (preprocessing included)
├── app/streamlit_app.py       GUI
├── docs/                      rubric checklist, sources, clarifications,
│                              team prompts, evidence pack, review outline
└── tests/                     12 focused tests
```

### How the notebooks are organised

Each algorithm gets its **own numbered section**: a plain-words explanation of
what it is, numbered steps for how it works, the settings worth tuning, a viva
note — then the code that trains *that one model* on our data, then its output.
All models share the single train/test split created once in §3, which is what
makes the comparison tables fair.

---

## 7. 🤖 AI-Assistance Disclosure

*Required by guidelines §7.5: "Generative AI tools may be used for code
scaffolding but not for analysis or interpretation. If AI assistance is used,
cite it in the README."*

**Tool used:** Claude (Anthropic), via the Claude interface, September 2026.

**What AI generated:**
* All Python module, script and notebook **scaffolding** in `src/`, `scripts/`,
  `tests/` and `app/`
* Plot-generation code and figure styling
* Structural documentation: repository layout, this README's structure, the
  rubric checklist, the review outline, and the guiding questions in
  `docs/team_analysis_prompts.md`
* Descriptions of **what the code does** and how each algorithm works
* The **draft** problem statements in §1, flagged as drafts for team rewriting

**What AI did NOT generate, and must not:**
* Any EDA observation or interpretation of any plot
* Any feature-engineering decision or its justification —
  `src/feature_engineering.py` ships **empty on purpose**
* Any model-selection argument or conclusion
* Any answer in `docs/team_analysis_prompts.md`

The notebooks mark these locations with **✍️ TEAM TO COMPLETE** cells containing
guiding questions and no answers.

**Metrics:** every number in §5 was computed by executing the code on the real
data. None was estimated, predicted or transcribed from an AI's expectation.
They can be reproduced with `python scripts/run_all.py --mode full` and
independently re-derived by `python scripts/validate_project.py`.

### Other source citations
* **scikit-learn** (BSD-3-Clause) — estimators, metrics, splitters, transformers
* **SciPy** (BSD-3-Clause) — statistical helpers
* No StackOverflow or blog code was pasted into this repository.

---

## 8. Outstanding team-authored sections

| # | Item | Rubric impact | Where |
|---|---|---|---|
| 1 | **Engineered feature — regression** | **B3 (1 mark)** | Q-RG3 |
| 2 | **Engineered feature — classification** | **B3** | Q-CF1 |
| 3 | EDA observations (both notebooks) | **A3 (1 mark)** | ✍️ cells |
| 4 | Final Part A model justification | **D2** | Q-CF2 |
| 5 | Duplicate-row policy | **B1** | Q-RG1 |
| 6 | Problem statements (final wording) | presentation | Q-X1 |
| 7 | Track conclusions | **E1 (1 mark)** + viva | ✍️ cells |

---

## 9. Limitations requiring team discussion

1. **Both datasets are synthetic.** Clean generating processes are why ten very
   different regression algorithms agree to within 1.2 % R². Do not read that as
   "the problem is solved."
2. **Regression: 127 duplicate rows, ~12.9 % feature-row overlap between train
   and test.** With five low-cardinality inputs this is arithmetically expected,
   not a pipeline bug — but decide it yourselves (Q-RG1, Q-RG2).
3. **Classification: 3.39 % positive class.** Weighted F1 is flattered by the
   majority class; the majority-class baseline already scores 0.9493. Failure
   recall is the metric that discriminates, and in Part A it ranges from 0.10 to
   0.38 untuned — and reaches 0.74 after tuning the Decision Tree.
4. **The tuned Decision Tree trades ranking quality for decision quality**
   (recall up, ROC-AUC down). Which you prefer depends on whether you deploy at a
   fixed threshold or rank machines for inspection.
5. **IC-1 is resolved** — the instructor confirmed teams select their own
   datasets. Four minor clarifications remain open; see
   `docs/instructor_clarifications.md`.

---

## 10. GUI status

| Item | Status |
|---|---|
| Interactive GUI accepting inputs, returning predictions | ✅ **Built** — `app/streamlit_app.py`, loads saved pipelines, runs locally |
| Public deployment | ❌ **Not done, not claimed.** No public URL exists. |

The app serves the **Review 1** models: the best regressor (Polynomial deg-2)
and the best **Part A** classifier (Decision Tree). It deliberately does not
load any Part B model.
