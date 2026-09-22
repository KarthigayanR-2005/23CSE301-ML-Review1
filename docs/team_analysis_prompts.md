# Team-Authored Analysis — Questions To Answer (Review 1)

*The clustering questions (Q-CL1 … Q-CL6) belong to Review 2 and live in that
repository.*

Guidelines §7.5: *"Each team's analysis, interpretation, and feature engineering
decisions must be original."* Generative AI may scaffold code but **not** the
analysis. Everything below is therefore deliberately unanswered.

Work through these **as a team**. Every member should be able to defend every
answer in the viva, not only the ones for their own track.

**Legend:** 🔴 blocks a rubric mark · 🟡 strengthens the review · ⚪ viva preparation

---

## Regression track

### 🔴 Q-RG1 — Duplicate-row policy
127 exact duplicate rows exist. Retaining them keeps 10,000 rows; removing them
leaves 9,873 (below the working preference).
- Given that the five inputs are low-cardinality integers and one binary flag,
  how many identical rows would you *expect* by chance in 10,000 draws?
- Coincidence, or a data-generation artefact?
- **Your decision and justification:**

### 🔴 Q-RG2 — Train/test overlap
`check_no_row_leakage` reports ~12.9 % of test rows whose feature values also
appear in training.
- Is this "leakage" in the harmful sense, or an inevitable consequence of low
  feature cardinality?
- Does it inflate the R² figures, and if so by how much would you guess?
- **Your position:**

### 🔴 Q-RG3 — Engineered feature *(rubric B3 — currently INCOMPLETE)*
`src/feature_engineering.py` is empty by design.
- Which feature will you create, from which columns?
- **State your justification before you measure it.**
- Register it, re-run `python scripts/run_all.py --track regression --mode full`,
  and report the honest before/after — including if it made things worse.
- **Feature / justification / result:**

### 🟡 Q-RG4 — Why do all ten models land within ~1 % R²?
Every model scores between about 0.977 and 0.989 test R².
- What does that say about the relationship between inputs and target?
- What does it say about how this synthetic data was generated?
- Why do the ensembles fail to beat plain linear regression here?

### 🟡 Q-RG5 — Model selection
- Which model do you nominate, and on what grounds — accuracy, stability,
  simplicity, interpretability, training cost?
- Tuning moved CV R² only in the fourth decimal. Does that change your pick?

### ⚪ Q-RG6 — Viva readiness
- What does R² = 0.989 actually mean in words?
- Why is RMSE ≈ 2.0 index points more useful to a reader than R²?
- What would a residual plot look like if the model were missing a non-linear term?

---

## Classification track

### 🔴 Q-CF1 — Engineered feature *(rubric B3 — currently INCOMPLETE)*
- Which physical quantity could you derive from the six inputs?
- **Justify before measuring.**
- Did it improve *failure recall*, and did it hurt anything else?
- **Feature / justification / result:**

### 🔴 Q-CF2 — Final Part A model selection *(rubric D2)*
- Which of the five do you select, and on which metric?
- Quote the before/after tuning numbers. If tuning did not help, say so plainly.
- Does your model trade failure recall for accuracy? Acceptable?

### 🟡 Q-CF3 — Which metric should lead, and why not accuracy?
Only 3.39 % of rows are failures; predicting "no failure" always scores ~96.6 %.
- Which metric leads your comparison and why?
- Would you lower the decision threshold below 0.5? What would that cost?

### 🟡 Q-CF4 — Why is failure recall low across most models?
Every untuned Part A model catches under 40% of failures; the best (Decision
Tree) catches 38%. Tuning lifted that to 74% — but cost ROC-AUC.
- Is that the models' fault, the imbalance, or the feature set?
- If you applied class weights or resampling, it must be inside training folds
  only. Did you? What happened?

### 🟡 Q-CF5 — The Naive Bayes assumption
Gaussian NB assumes features are conditionally independent given the class. The
correlation heatmap shows at least one strongly correlated pair.
- Name the pair and the physical reason.
- How did the violation show up in GaussianNB's score?

### 🟡 Q-CF5b — The tuning trade-off *(new, and worth the most viva credit)*
Tuning the Decision Tree moved failure recall 0.38 → 0.74 while ROC-AUC fell
0.92 → 0.88.
- How can a model get BETTER at deciding and WORSE at ranking at the same time?
- Which would you deploy, and does your answer depend on whether you act at a
  fixed threshold or rank machines for inspection?

### ⚪ Q-CF6 — Viva readiness
- Why is ROC-AUC computed from probabilities and not from predicted labels?
- What does weighted F1 average over, and why is it flattering here?
- Read the top two splits of the decision tree: what rule has it learned?

---

## Cross-cutting

### 🟡 Q-X1 — Problem statements
`README.md` carries three **draft** problem statements. Rewrite them in your own
words — they are drafts, not final text.

### 🟡 Q-X2 — Both datasets are synthetic
- What can you legitimately conclude from a model trained on synthetic data?
- What would you need before trusting either in the real world?

### ⚪ Q-X3 — Presentation story arc *(rubric E1)*
problem → data → method → results → conclusion. Who presents which track, and
what is the single sentence each track must land?
