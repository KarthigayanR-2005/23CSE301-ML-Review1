# Evidence Pack — the numbers for every ✍️ cell (Review 1)

*Clustering evidence (Q-CL1 … Q-CL6) lives in the Review 2 repository.*

**What this is:** every figure you need, already extracted, so you are not
digging through 47 tables while writing.

**What this is not:** the answers. Guidelines §7.5 requires the analysis to be
your own, and the viva is worth 10 of your 50 marks — answers you didn't write
are answers you can't defend when the examiner picks you and asks "why?".

**How to use it:** each block gives you the FACTS and then the ONE judgment
you have to supply. Most cells need two or three sentences, not an essay.

---

# REGRESSION

## Q-RG1 · Duplicate rows — the decision

**Facts:**
- 127 exact duplicate rows in 10,000.
- Input cardinality: Hours Studied **9** values, Previous Scores **60**, Sleep Hours **6**, Papers **10**, Extracurricular **2**.
- Distinct input combinations: 9 × 60 × 6 × 10 × 2 = **64,800**.
- Draw 10,000 rows from 64,800 possible input patterns and repeats are arithmetically unavoidable (birthday-problem territory).
- Target takes 91 distinct values, so full-row combinations ≈ 5.9 million.
- Nothing else is wrong: **0 missing cells**, no impossible values.

**You decide:** coincidence or data-collection artefact? Keep or drop? One
sentence of reasoning. (Dropping gives 9,873 rows — under the 10,000 preference.)

## Q-RG2 · Train/test overlap

**Facts:**
- 258 of 2,000 test rows (**12.9%**) have feature values that also appear in training.
- With only 64,800 possible input patterns and 8,000 training rows, roughly 12% of any new sample will match a training pattern by chance.
- These are *feature* matches. The target still varies.

**You decide:** is this harmful leakage, or unavoidable given the cardinality?
Does it inflate R²?

## Q-RG3 · Engineered feature ← **required, rubric B3, 1 mark**

**Facts to work from:**
- Correlation with target: Previous Scores **0.915**, Hours Studied **0.374**, Sleep Hours **0.048**, Papers **0.043**.
- Max correlation *between* inputs: **0.018** — the inputs are essentially independent of each other.
- Lasso zeroed **0 of 5** coefficients — every input carries signal.
- Current best test R² = **0.98899**. Ceiling is close.

**You decide:** which feature, built from which columns, and **why you expect it
to help — written before you measure**. Then register it in
`src/feature_engineering.py`, re-run, and report the honest result.

## Q-RG4 · Why all ten models land within 1.2% R²

**Facts:**
- Spread: 0.9768 (KNN) to 0.9890 (Polynomial). Range = 0.0122.
- Linear models occupy ranks 1–5; every ensemble and kernel method ranks below them.
- Random Forest: train R² **0.9975** vs test **0.9862** — a 0.011 gap.
- Polynomial degree 3 (55 terms) scored **worse** than degree 2 (20 terms).
- Dataset is **synthetic**.

**You decide:** what does this pattern say about how the data was generated, and
why flexibility doesn't pay here?

## Q-RG5 · Model selection

**Facts:**
- Rank 1 Polynomial deg-2: R² 0.988989, 20 terms, 0.016 s.
- Rank 2 Linear: R² 0.988983, 5 terms, 0.014 s. **Difference: 0.000006.**
- Rank 7 Random Forest: R² 0.986175, 300 trees, 3.426 s.
- Tuning moved CV R² in the 5th decimal.

**You decide:** which do you ship, and on what grounds — accuracy, simplicity,
interpretability, cost?

---

# CLASSIFICATION

## Q-CF1 · Engineered feature ← **required, rubric B3**

**Facts:**
- Mechanical power = torque × angular velocity. You have both: Torque [Nm] and Rotational speed [rpm].
- Correlation with Target: Torque **0.191**, Tool wear **0.105**, Air temp **0.083**, Process temp **0.036**, Speed **−0.044**. No single feature is strong.
- Failure-vs-healthy means — Torque **50.2 vs 39.6**, Tool wear **143.8 vs 106.7**, Speed **1496 vs 1540**, Air temp **300.9 vs 300.0**.
- Temperature *difference* between process and air averages ~10 K.

**You decide:** which derived quantity, and **your justification, written first**.
Then measure the effect on failure recall and report it honestly either way.

## Q-CF2 · Final Part A model selection ← **required**

**Facts:**

| Model | F1w | Recall(fail) | Precision(fail) | ROC-AUC | CV F1w |
|---|---|---|---|---|---|
| A4 Decision Tree | **0.9714** | **0.3824** | 0.7879 | 0.9219 | **0.97480** |
| A2 KNN | 0.9679 | 0.2941 | 0.8333 | 0.8291 | 0.96786 |
| A5 SVC | 0.9635 | 0.2059 | **0.8750** | **0.9468** | 0.96677 |
| A1 Logistic | 0.9560 | 0.1029 | 0.6364 | 0.8994 | 0.96176 |
| A3 Gaussian NB | 0.9506 | 0.1176 | 0.2500 | 0.8468 | 0.95530 |

- Baseline: accuracy 0.9660, F1w 0.9493.
- Different metrics crown different winners: F1w and recall → Decision Tree, AUC → SVC, precision → SVC.
- **After tuning**, the Decision Tree reaches recall **0.7353** and F1w **0.98421** — but ROC-AUC drops to **0.8847**.

**You decide:** which model, which metric drove it, and is the recall/precision
trade acceptable for a maintenance setting? Does the tuned or untuned version
win?

## Q-CF3 · Which metric should lead

**Facts:**
- 339 failures in 10,000 rows = **3.39%**.
- Predicting "no failure" always → accuracy **0.9660**, F1w **0.9493**.
- Accuracy across the five Part A models spans only 0.958–0.976.
- Recall on failures spans **0.103–0.382** untuned in Part A — a nearly 4x spread, and **0.735** once the Decision Tree is tuned.
- Logistic Regression: accuracy 0.9675 (above baseline) while catching **7 of 68** failures.

**You decide:** which metric leads, and would you move the 0.5 threshold?

## Q-CF5 · The Naive Bayes assumption

**Facts:**
- Air temp ↔ Process temp: **r = 0.876**.
- Rotational speed ↔ Torque: **r = −0.875**.
- GaussianNB: F1w **0.9506** — last of the five; precision on failures **0.25**.
- It multiplies per-feature likelihoods as if independent.

**You decide:** name the pair, the physical reason, and connect it to the score.

## Q-CF6 · Tree structure

**Facts:** `results/figures/classification/clf_decision_tree.png`, depth-3 view.
Logistic odds ratios: Torque **15.2×**, Speed **7.5×**, Air temp **4.2×**,
Tool wear **2.2×** per standard deviation.
Decision Tree importances: Torque **0.515**, Air temp **0.147**, Tool wear
**0.123**, Speed **0.111**, Process temp **0.100**, Type **0.005**.

**You decide:** read the top two splits and state the physical rule in words.

---

# CROSS-CUTTING

## Q-X2 · Both datasets are synthetic

**Facts:** regression — 10 algorithms within 1.2% R², max inter-feature
correlation 0.018, no missing values. Classification — exactly 3.39% positives,
two feature pairs at |r| ≈ 0.88. Real-world data rarely looks this tidy.

**You decide:** what can you legitimately conclude, and what would you need
before trusting either in production?
