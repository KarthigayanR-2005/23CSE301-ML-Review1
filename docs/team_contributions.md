# Team Contributions — Review 1

**Names are intentionally blank** — fill them in yourselves.

Guidelines §7.4: divide by **track**, not by step. Do **not** split it as "one
person does EDA, one does models, one does slides." Each owner must be able to
explain *every line* of their track — and the instructor may ask **any member
about any part** during the viva.

---

## Primary ownership

| Track | Primary owner | Cross-reviewer | Second reviewer |
|---|---|---|---|
| Regression (10 algorithms) | _______________ | _______________ | _______________ |
| Classification Part A (5 algorithms) | _______________ | _______________ | _______________ |
| EDA, docs and presentation | _______________ | _______________ | _______________ |

*(Review 2 ownership — Part B and clustering — is tracked in that repository.)*

Suggested rotation so every member reviews work they do not own:

| Member | Owns | Reviews |
|---|---|---|
| Member 1: ______ | Regression | Classification Part A |
| Member 2: ______ | Classification Part A | EDA + docs |
| Member 3: ______ | EDA, docs, presentation | Regression |

## Shared responsibilities

| Item | Owner | Status |
|---|---|---|
| README problem statements (final wording) | _______ | ⬜ draft only |
| Feature engineering — regression (Q-RG3) | _______ | 🔴 not started |
| Feature engineering — classification (Q-CF1) | _______ | 🔴 not started |
| All notebook "TEAM TO COMPLETE" cells | all three | 🔴 unwritten |
| Final Part A model justification (Q-CF2) | _______ | 🔴 not started |
| Instructor clarifications (IC-2, IC-4) | _______ | ⬜ to raise (IC-1 resolved) |
| Review 1 slides | _______ | ⬜ |
| Streamlit app demo | _______ | ✅ built, local only |

## What "ownership" means for the viva

The Review 1 viva is worth **5 of 25 marks**, and the instructor may direct any
question to any member about any part of the work.

Before the review, every member should be able to, for **both** tracks:

1. Explain the dataset, its target, and why columns were excluded.
2. Explain why *that* algorithm suits *that* problem.
3. Interpret every metric in the tables — in plain words, not formulas.
4. Justify preprocessing, and explain what data leakage is and how the pipelines prevent it.
5. Explain the trade-offs behind the final model choice.
6. Say honestly what the project does **not** establish.

**Rehearsal method:** each member is quizzed on the track they do **not** own.

The highest-value question to rehearse: *tuning the Decision Tree raised failure
recall from 0.38 to 0.74 but lowered ROC-AUC from 0.92 to 0.88 — how, and which
version would you ship?*

## Commit hygiene

The guidelines require a meaningful commit history, not a single bulk upload.
Commit at each real milestone, with a message saying what changed — and each
member should commit their own ✍️ answers under their own name.
