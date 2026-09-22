# Review 1 — Presentation Outline

**Scope:** full Regression track + Classification Part A · **25 marks**
**Repository:** this one. Review 2 work is in a separate repository.
In-person, all three members present, any member may be asked about any part.

> This is a **structure**, not a script. The content of every "say this" slot is
> yours to write — that is what rubric E1 and the viva assess.

---

## Suggested timing (~12 min + questions)

| # | Slide | Min | Owner |
|---|---|---:|---|
| 1 | Title, team, tracks | 0.5 | — |
| 2 | Problem statements ×2 | 1.0 | — |
| 3 | Datasets, shapes, synthetic disclosure | 1.0 | — |
| 4 | Regression EDA | 1.5 | regression owner |
| 5 | Preprocessing & leakage prevention | 1.5 | regression owner |
| 6 | Regression: 10-model table | 2.0 | regression owner |
| 7 | CV + tuning | 1.0 | regression owner |
| 8 | Residual + pred-vs-actual + importance | 1.0 | regression owner |
| 9 | Classification Part A: EDA + imbalance | 1.5 | classification owner |
| 10 | Part A: 5 models + confusion matrices + tuning | 2.0 | classification owner |
| 11 | Honest limitations + next steps | 0.5 | third member |

---

## Slide notes

**3 — Datasets.** State plainly that both supervised datasets are **synthetic**.
Say it before the results, not after — it frames why the numbers are so high and
shows you understand what they mean.

**5 — Preprocessing.** This is where marks are won cheaply. Show one pipeline
diagram and say: *scalers are fitted inside the pipeline, so they refit on each
CV fold and never see the test set.* §7.1 makes full-data fitting a deduction —
demonstrate you avoided it. Mention `Failure Type` exclusion here (target
leakage) even though it is the classification track; examiners like it.

**6 — Regression table.** Use `results/tables/regression_comparison.csv`.
Anticipate the obvious question: *why are all ten models within 1.2 % R²?*
(Q-RG4.) Have an answer ready.

**7 — Tuning.** Be upfront: the gains are in the fifth decimal. Saying so is
stronger than dressing it up. Explain **why** — models this close to the ceiling
have little left to tune.

**9 — Imbalance.** Lead with the majority-class baseline: **accuracy 0.9660,
weighted F1 0.9493**. Any model score must be read against that. This single
slide prevents the worst viva outcome — quoting 97 % accuracy as a success.

**10 — Part A.** Point at the failure-recall column, not accuracy. Logistic
Regression catches **10 %** of failures at 96.75 % accuracy. That contrast is the
most instructive thing in the whole track.

Then show the tuning row. The Decision Tree went from catching **38 %** of
failures to **74 %** — and its ROC-AUC *fell* from 0.92 to 0.88. Volunteering
that trade-off, and explaining it (better decisions at one threshold, worse
ranking across all thresholds), is the single strongest thing you can do in this
review.

**11 — Limitations.** Say what is unfinished, including feature engineering if it
is still outstanding. Do not claim work you have not done.

---

## Pre-review checklist

- [ ] All ✍️ cells in `regression.ipynb` written
- [ ] ✍️ cells in `classification.ipynb` written (all of it — the notebook is Part A only)
- [ ] **Engineered feature registered + justified** (rubric B3 — Q-RG3/Q-CF1)
- [ ] Duplicate-row policy decided (Q-RG1)
- [ ] Notebooks re-executed after the above
- [ ] `python scripts/validate_project.py` — no FAIL rows
- [ ] Git history shows milestone commits, not one bulk upload
- [x] ~~IC-1 raised~~ — **resolved: teams select their own datasets**
- [ ] IC-2 raised (should the regression split be stratified?)
- [ ] Each member quizzed on a track they do not own

## Questions to expect

1. Why is the scaler inside the pipeline instead of applied first?
2. What is R² = 0.989 in plain words?
3. Why does Random Forest have train R² 0.9975 but test 0.9862?
4. Why not stratify the regression split?
5. Why exclude `Failure Type`? What would including it have done?
6. Your model has 96.75 % accuracy and catches 10 % of failures. Is it good?
7. Which feature did you engineer, and why did you expect it to help?
8. Your tuned tree catches twice as many failures but has a lower ROC-AUC. Explain.
9. Why is Gaussian Naive Bayes last, and which assumption does this data break?
