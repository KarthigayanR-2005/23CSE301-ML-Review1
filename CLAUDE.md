# CLAUDE.md — Project Rules and Progress (Review 1)

Read this first in any later session. It records the **rules** this project runs
under and the **actual** state of the work.

---

## 0. Scope — do not widen it

This repository is the **Review 1 deliverable only**:

* full regression track (10 algorithms)
* classification **Part A** (5 algorithms)

Classification Part B and the clustering track belong to **Review 2**, in a
separate repository. Do **not** add them here. `scripts/validate_project.py`
checks that they stay absent, and `tests/` asserts `build_part_b` does not
exist.

---

## 1. Hard rules (do not violate)

### Academic integrity (guidelines §7.5)
AI may generate code scaffolding. AI must **NOT** generate:
- EDA observations or plot interpretations
- Feature-engineering decisions or their justifications
- Model-selection arguments
- Conclusions

`src/feature_engineering.py` is **empty on purpose**. Do not populate it with an
invented feature. Do not answer anything in `docs/team_analysis_prompts.md`.
Do not fill any `✍️ TEAM TO COMPLETE` notebook cell.

### Honesty
- Never fabricate a result, improvement, execution status, or commit.
- If tuning does not help, report that it did not.
- If tuning helps one metric and hurts another, report **both**. (This already
  happened: tuning the Decision Tree moved failure recall 0.38 → 0.74 while
  ROC-AUC fell 0.92 → 0.88. Both numbers are in the README.)
- Distinguish **implemented** / **executed** / **awaiting team input**.
- Never mark the project submission-ready while team sections are outstanding.

### Data
- `data/raw/` is immutable. No script writes there.
- Never substitute a dataset or manufacture rows to work around a problem.
- Never commit credentials. Kaggle tokens live in `~/.kaggle/kaggle.json`.

### Methodology
- `random_state=42` everywhere.
- Split before any data-driven decision.
- All transformers inside Pipelines → refitted per CV fold.
- Select models on **training** CV; the held-out set is scored once.
- Never scale the regression target.

---

## 2. Architecture

One implementation, in `src/`. Notebooks and `scripts/run_all.py` both import
it, so they cannot drift apart. Do not copy pipeline logic into a notebook.

```
src/config.py                 paths, seed, dataset contracts (edit specs here)
src/data_loading.py           load + audit + provenance/checksums
src/preprocessing.py          splits, ColumnTransformers, leakage checks
src/feature_engineering.py    EMPTY HOOK - team input required
src/evaluation.py             metrics + comparison tables + persistence
src/plotting.py               every figure (styling rules enforced here)
src/regression_models.py      10 regressors + tuning grids
src/classification_models.py  5 Part A classifiers + tuning grids
scripts/algorithm_content.py  per-algorithm write-ups used by the notebooks
```

Adding a model = add it to the right `build_*` function and, if tunable, to
`PARAM_GRIDS`. Nothing else changes; the tables and plots pick it up.

The notebooks are **generated**, not hand-edited for structure: `build_notebooks.py`
turns each entry in `algorithm_content.py` into a section (markdown explanation
→ code that trains that one model → its output). Hand-written ✍️ answers live in
the generated `.ipynb` files, so **re-running `build_notebooks.py` overwrites
them** — commit your answers first, or paste them back in.

---

## 3. Commands

```bash
python scripts/download_data.py        # verify/acquire data
python scripts/run_all.py --mode full  # the reported run (~1 min)
python scripts/run_all.py --mode dev   # fast pass; outputs suffixed _dev
python scripts/build_notebooks.py      # regenerate notebooks from source
python scripts/execute_notebooks.py    # run them top-to-bottom
python -m pytest tests/ -v
python scripts/validate_project.py     # live rubric + integrity status
streamlit run app/streamlit_app.py
```

**`--mode dev` results are never the reported figures.** They carry a `_dev`
suffix and the run prints a DEVELOPMENT RUN banner.

---

## 4. Progress log

### Completed and executed
- [x] Both datasets verified; shapes match the contract exactly
- [x] Provenance + SHA-256 checksums recorded
- [x] Regression: 10 models, CV, 2× GridSearchCV, all required figures
- [x] Classification Part A: 5 models, comparison table, confusion matrices,
      ROC curves, CV, 2× GridSearchCV, odds, tree viz, importances, KNN metrics
- [x] Both notebooks executed top-to-bottom with outputs, 0 errors
- [x] 12 focused tests passing
- [x] Streamlit app serving the Review 1 models (local only — NOT deployed)
- [x] README with computed numbers + AI disclosure

### Outstanding — team only
- [ ] **Feature engineering, both tracks** (rubric B3) — Q-RG3, Q-CF1
- [ ] All EDA observations (rubric A3) — ✍️ cells
- [ ] Final Part A model justification (rubric D2) — Q-CF2
- [ ] Duplicate-row policy (Q-RG1)
- [ ] Track conclusions (rubric E1)
- [ ] Raise IC-2 and IC-4 with the instructor

---

## 5. Known issues and decisions already taken

| Item | Decision | Revisit? |
|---|---|---|
| 127 duplicate regression rows | **retained**; dedup would give 9,873 rows | Q-RG1 |
| ~12.9 % train/test feature-row overlap | reported, not "fixed" | Q-RG2 |
| Regression stratification | not stratified; binned option available | IC-2 |
| IC-1 (were datasets assigned?) | **RESOLVED** — teams choose their own | — |
| Decision Tree tuning | recall up, ROC-AUC down — both reported | Q-CF5b |
| Public deployment | **not done, not claimed** | — |

---

## 6. Environment

Python 3.11+ (built on 3.12, verified on 3.13). scikit-learn 1.8.0,
pandas 3.0.2, numpy 2.4.4. Note `sklearn.metrics.root_mean_squared_error` is
used (the `squared=False` argument was removed in newer scikit-learn). Full run
≈ 1 minute.

`src/` contains no Unix-only imports — `memory_probe` and its `resource`
dependency belonged to the clustering track and are not present here, so the
Windows compatibility issue that affected the full project does not arise.
