"""build_notebooks.py - generate notebooks/{regression,classification}.ipynb

Structure: EVERY algorithm gets its own numbered section inside the notebook -

    Markdown : what it is (plain words), how it works (numbered steps),
               settings you can tune, what to say in the viva
    Code     : that ONE model built, trained on the data, and evaluated,
               with its own visible output
    Extras   : the algorithm-specific display the PDF asks for (coefficients,
               Lasso sparsity, degree comparison, tree plot, odds ratios, ...)

Every model trains on the SAME split created once in the preprocessing section,
so the consolidated comparison tables required by rubric C2 / A2 stay fair.

The algorithm write-ups are imported from scripts/algorithm_content.py, so
there is one source of truth for them.

Scope: REVIEW 1 - the full regression track and classification PART A only.

    python scripts/build_notebooks.py          # write the .ipynb files
    python scripts/execute_notebooks.py        # run them top-to-bottom
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import nbformat as nbf

from src import config
from algorithm_content import CLASSIFICATION, REGRESSION

NB_DIR = config.NOTEBOOKS_DIR


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str):
    return nbf.v4.new_code_cell(text.strip("\n"))


def team(heading: str, *questions: str):
    """Retired: the team-observation placeholders were removed from the
    notebooks. Kept as a no-op so existing call sites stay valid; it returns
    None and `write()` filters those out."""
    return None


def show(var: str = "p") -> str:
    return f"from IPython.display import Image, display\ndisplay(Image(filename=str({var})))"


def algo_markdown(m: dict, section: str) -> str:
    steps = "\n".join(f"{i}. {s}" for i, s in enumerate(m["how"], 1))
    return f"""
### {section} {m["title"]}

**What it is**

{m["what"]}

**How it works**

{steps}

**Settings you can tune**

{m["settings"]}

**Viva note** - {m["viva"]}
"""


def strip_pipeline_import(imports: str) -> str:
    """Pipeline is imported once in the setup cell."""
    return "\n".join(l for l in imports.splitlines()
                     if "sklearn.pipeline" not in l).strip()


HEADER = """
# 23CSE301 Machine Learning - Capstone Project
## {title}

**Track owner:** _(name - see `docs/team_contributions.md`)_
**Review:** {review}
**Dataset:** {dataset}
**Source:** {url}

---

### How to read this notebook

Each algorithm has its **own section**: first a short explanation of what it is
and how it works, then the code that builds and trains *that* model on our data,
then its result. Every model uses the same train/test split created in section 3,
which is what makes the comparison tables fair.

Pipeline steps come from `src/`, the same code `scripts/run_all.py` runs, so the
notebook and the scripts cannot drift apart.

**AI assistance:** code scaffolding was AI-generated (see `README.md`,
AI-Assistance Disclosure).
"""

SETUP = """
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))          # project root on the path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

from src import config, data_loading as dl, preprocessing as pp
from src import feature_engineering as fe, evaluation as ev, plotting as pl

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)
warnings.filterwarnings("default")                  # warnings stay visible

print("random_state =", config.RANDOM_STATE,
      "| test_size =", config.TEST_SIZE,
      "| cv folds =", config.CV_FOLDS)
"""


# ==========================================================================
def regression_nb():
    spec = config.REGRESSION
    cells = [
        md(HEADER.format(title="Regression Track - Student Performance",
                         review="Review 1 (full track)",
                         dataset=f"{spec['name']} - `{spec['filename']}`",
                         url=spec["url"])),
        md("## 0. Setup"),
        code(SETUP),
        code(f"""
TARGET = {spec["target"]!r}
NUM = {spec["numeric_features"]!r}
CAT = {spec["categorical_features"]!r}
print("target          :", TARGET)
print("numeric inputs  :", NUM)
print("categorical inpt:", CAT)
"""),

        md("""
## 1. Data loading and audit  *(Rubric A1)*

The raw CSV is read from `data/raw/` and never modified. We report shape,
dtypes, missing-value counts and the target distribution.
"""),
        code("""
raw = dl.load_raw("regression")
summary = dl.audit_summary(raw, "regression")
for k, v in summary.items():
    print(f"  {k}: {v}")
"""),
        code("""dl.audit(raw, "regression", "raw")"""),
        code("""raw.head()"""),
        code("""raw.describe(include="all").T"""),

        md("""
### 1.1 Duplicate rows - a decision for the team

127 exact duplicate rows exist. All five inputs are low-cardinality integers or
a binary flag, so identical rows arise by chance in a 10,000-row synthetic
sample. Removing them leaves 9,873 rows, **below the 10,000-row working
preference**. The reported run **retains** them; both options are measured below.
"""),
        code("""
df = dl.prepare("regression", raw)
dedup = pp.drop_exact_duplicates(df)
print(f"\\nretained : {len(df)} rows  (the reported configuration)")
print(f"if dropped: {len(dedup)} rows  -> below the 10,000 preference: {len(dedup) < 10000}")

combos = 1
for c in NUM + CAT:
    combos *= df[c].nunique()
print(f"\\ndistinct input combinations possible: {combos:,}")
print(f"rows drawn from those combinations   : {len(df):,}")
"""),
        team("Duplicate-handling decision (Q-RG1)",
             "Given that many distinct input combinations, how many identical rows would you EXPECT by chance in 10,000 draws?",
             "Coincidence, or a data-collection artefact?",
             "Which option do you choose, and why?"),

        md("""
## 2. Exploratory data analysis  *(Rubric A2, A3)*

Distribution plot per input feature, correlation heatmap, target distribution,
and two feature-target scatter plots.
"""),
        code("""
p = pl.distribution_grid(df[NUM + CAT],
                         "Student Performance - input feature distributions",
                         "nb_reg_feature_distributions", "regression")
"""),
        code(show()),
        team("Feature distributions",
             "Which features are roughly uniform and which are skewed?",
             "Does any feature show an implausible value or a suspicious spike?",
             "What does each shape imply for scaling, or for models that assume normality?"),

        code("""
p = pl.correlation_heatmap(df, "Student Performance - correlation heatmap (numeric)",
                           "nb_reg_correlation_heatmap", "regression")
"""),
        code(show()),
        code("""
corr_t = df.select_dtypes("number").corr()[TARGET].drop(TARGET)
print("correlation of each input with the target:")
print(corr_t.round(4).sort_values(ascending=False).to_string())
"""),
        team("Correlation structure",
             "Which input correlates most strongly with Performance Index, and by how much does it lead the others?",
             "Is any correlation BETWEEN inputs high enough to worry about multicollinearity in the linear models?",
             "Does the ranking match what you would expect from the domain?"),

        code("""
p = pl.target_distribution(df[TARGET], "Target distribution - Performance Index",
                           "nb_reg_target_distribution", "regression")
"""),
        code(show()),
        code("""
top2 = corr_t.abs().sort_values(ascending=False).index[:2].tolist()
print("two strongest relationships with the target:", top2)
p = pl.feature_target_scatter(df, top2, TARGET,
                              "nb_reg_feature_target_scatter", "regression")
"""),
        code(show()),
        team("Feature-target relationships",
             "Is each relationship linear, curved, or absent?",
             "How much vertical spread is there at a fixed x - how much variance can that feature alone NOT explain?",
             "Does this support a linear model as the baseline?"),

        md("""
## 3. Preprocessing and the held-out split  *(Rubric B1, B2)*

The split happens **before** any modelling decision, so the test set stays
genuinely unseen.

**Why preprocessing goes inside a Pipeline.** If we scaled the whole dataset
first, the scaler would have computed its mean and standard deviation from the
test rows too - that is data leakage, and section 7.1 makes it a mark deduction.
Inside a `Pipeline`, `.fit()` fits the scaler on training data only, and during
cross-validation it is refitted from scratch on every fold.

**Stratification note.** Rubric B2 asks for a stratified split, but
stratification needs discrete classes and our target is continuous. We use an
unstratified random split and log the ambiguity as **IC-2**.
`pp.split_supervised(..., stratify_bins=k)` implements quantile-binned
stratification if the instructor confirms it is wanted.
"""),
        code("""
X_train, X_test, y_train, y_test = pp.split_supervised(df, TARGET, stratify=False)
print(f"\\ntrain: {X_train.shape}   test: {X_test.shape}")
print("overlap check:", pp.check_no_row_leakage(X_train, X_test))
"""),
        code("""
# What the preprocessor actually does, shown once.
prep = pp.make_preprocessor(NUM, CAT, scale=True)
prep.fit(X_train)                                   # TRAINING data only
print("output feature names:", list(prep.get_feature_names_out()))
print("\\nnumeric block   : median impute -> StandardScaler")
print("categorical block: most-frequent impute -> OneHotEncoder(drop='if_binary')")
Xt = prep.transform(X_train)
print("\\ntraining data after transform - mean ~0, std ~1:")
print("  mean:", np.round(Xt[:, :len(NUM)].mean(axis=0), 6))
print("  std :", np.round(Xt[:, :len(NUM)].std(axis=0), 6))
"""),
        team("Train/test overlap (Q-RG2)",
             "Why is the overlap percentage this large here, and is it 'leakage' in the harmful sense?",
             "Would deduplicating before splitting change your answer?"),

        md("""
### 3.1 Feature engineering  *(Rubric B3)* - NOT YET SATISFIED

`src/feature_engineering.py` ships **empty on purpose**. Until the team registers
a feature *and its justification*, the baseline pipeline runs and this rubric
item reports INCOMPLETE.
"""),
        code("""fe.status("regression")"""),
        team("Engineered feature (Q-RG3) - required for rubric B3",
             "Which new feature will you create, from which existing columns?",
             "Why do you expect it to help? Write this BEFORE you measure it.",
             "After registering it in src/feature_engineering.py and re-running, did it help? Report honestly either way."),

        md("""
## 4. The ten regression algorithms  *(Rubric C1)*

Each gets its own section below: what it is, how it works, the code that trains
it on our data, and its result. All ten use the same `X_train` / `X_test` from
section 3.

The helper below records each result so section 5 can assemble the comparison
table.
"""),
        code("""
results = {}     # model name -> metrics dict
fitted  = {}     # model name -> fitted Pipeline

def record(name, model):
    \"\"\"Evaluate a fitted pipeline on the held-out test set and store it.\"\"\"
    m = ev.regression_metrics(y_test, model.predict(X_test))
    m["Train_R2"] = ev.regression_metrics(y_train, model.predict(X_train))["R2"]
    results[name] = m
    fitted[name] = model
    print(f"{name}")
    print(f"  R2   : {m['R2']:.6f}")
    print(f"  RMSE : {m['RMSE']:.4f}   (Performance Index points)")
    print(f"  MAE  : {m['MAE']:.4f}   (Performance Index points)")
    print(f"  train R2 (overfitting check): {m['Train_R2']:.6f}")
    return m
"""),
    ]

    for i, m in enumerate(REGRESSION, start=1):
        cells.append(md(algo_markdown(m, f"4.{i}")))
        cells.append(code(f"""{strip_pipeline_import(m["imports"])}

model = Pipeline([
    ("prep", pp.make_preprocessor(NUM, CAT, scale={m['scale']})),
{m['poly']}    ("model", {m['estimator']}),
])

model.fit(X_train, y_train)
record("{m['title']}", model)
"""))
        if m["extra"].strip():
            cells.append(code(m["extra"].strip()))

    cells += [
        team("Coefficients, sparsity and polynomial degree",
             "State in one sentence, in plain language, what the largest Linear Regression coefficient means for a student.",
             "Do the signs of all coefficients match your domain expectation?",
             "How many coefficients did Lasso drive to zero, and what does that imply about redundant features?",
             "Compare train R2 and test R2 as the polynomial degree rises. At which degree does overfitting begin, and how do you see it?"),

        md("""
## 5. Comparative evaluation  *(Rubric C2)*

One table, all ten models, same test split, ranked by R2.
"""),
        code("""
table = (pd.DataFrame(results).T
         .reset_index().rename(columns={"index": "Model"})
         .sort_values("R2", ascending=False).reset_index(drop=True))
table.index = table.index + 1
table.index.name = "Rank"
table = table[["Model", "R2", "RMSE", "MAE", "Train_R2"]]
print(f"{len(table)} models compared on the same held-out split\\n")
table
"""),
        code("""
p = pl.model_comparison_bar(table, "R2", "Regression models ranked by test R2",
                            "nb_reg_model_comparison", "regression")
"""),
        code(show()),
        team("Comparative results",
             "Which model ranks first, and by how much does it beat the plain linear baseline?",
             "Is that margin large enough to matter, given RMSE is in Performance-Index points?",
             "What does it tell you that regularised linear models and tree ensembles land so close together?",
             "Which model shows the largest gap between train R2 and test R2, and what is that gap called?"),

        md("""
## 6. Cross-validation  *(mandatory: 5-fold CV R2 for the two best models)*

Leading models are nominated using **training-side** cross-validation. The
held-out test set plays no part in selection.

**How 5-fold CV works:** split the training data into 5 equal parts; train on 4
and score on the 1 left out; repeat 5 times so each part is held out once;
average the 5 scores. Because the whole Pipeline is passed in, the scaler is
refitted inside every fold.
"""),
        code("""
cv_all = ev.cv_scores(fitted, X_train, y_train, scoring="r2")
cv_all.drop(columns="fold_scores").sort_values("CV_mean", ascending=False)
"""),
        code("""
leaders = cv_all.sort_values("CV_mean", ascending=False)["Model"].head(2).tolist()
print("two leading models, selected on TRAINING cross-validation:")
for l in leaders:
    print("   ", l)
cv_all[cv_all["Model"].isin(leaders)][["Model", "CV_mean", "CV_std", "fold_scores"]]
"""),

        md("""
## 7. Hyperparameter tuning  *(Rubric C3)*

`GridSearchCV` on at least two models. It tries every combination in the grid,
scoring each by cross-validation on the training data only, and keeps the best.

**If tuning does not improve a model, that is what gets reported.**
"""),
        code("""
from src import regression_models as rm

tunable = [m for m in leaders if m in rm.PARAM_GRIDS]
for cand in rm.PARAM_GRIDS:                       # ensure at least two
    if len(tunable) >= 2:
        break
    if cand not in tunable:
        tunable.append(cand)
print("tuning:", tunable)
for t in tunable:
    print(f"  grid for {t}: {rm.PARAM_GRIDS[t]}")
"""),
        code("""
rows, tuned = {}, {}
for name in tunable:
    before_cv   = float(cv_all.loc[cv_all["Model"] == name, "CV_mean"].iloc[0])
    before_test = float(table.loc[table["Model"] == name, "R2"].iloc[0])
    gs = rm.tune(name, fitted[name], X_train, y_train)
    after = ev.regression_metrics(y_test, gs.best_estimator_.predict(X_test))
    rows[name] = {
        "best_params": gs.best_params_,
        "CV_R2_before": round(before_cv, 6),
        "CV_R2_after": round(gs.best_score_, 6),
        "HeldOut_R2_before": round(before_test, 6),
        "HeldOut_R2_after": round(after["R2"], 6),
        "improved_on_CV": bool(gs.best_score_ > before_cv),
    }
    tuned[name] = gs.best_estimator_
pd.DataFrame(rows).T
"""),
        team("Tuning outcome",
             "Did tuning change the score meaningfully, or only in the fourth or fifth decimal place?",
             "What does the size of the change suggest about whether these models were under-regularised to begin with?",
             "Would you ship the tuned or the default configuration, and why?"),

        md("""
## 8. Required visualisations  *(Rubric C4)*

**How to read a residual plot.** Residual = actual - predicted. Points should
scatter randomly around the zero line with roughly constant spread. A funnel
shape means the error grows with the prediction; a curve means a non-linear term
is missing.
"""),
        code("""
best_name = table.sort_values("R2", ascending=False).iloc[0]["Model"]
best_model = tuned.get(best_name, fitted[best_name])
y_pred = best_model.predict(X_test)
print("best model on the held-out set:", best_name)

p1 = pl.residual_plot(y_test, y_pred, best_name, "nb_reg_residuals", "regression")
p2 = pl.predicted_vs_actual(y_test, y_pred, best_name, "nb_reg_pred_vs_actual", "regression")
"""),
        code(show("p1")),
        code(show("p2")),
        team("Residual diagnostics",
             "Are residuals centred on zero with roughly constant spread, or do they fan out?",
             "Does the residual histogram look approximately normal?",
             "What would a visible curve here tell you about a missing feature or a missing non-linear term?"),
        code("""
imp = rm.tree_importances(fitted["7. Random Forest Regressor"])
p = pl.importance_plot(imp.values, imp.index,
                       "Random Forest Regressor - feature importance",
                       "nb_reg_feature_importance", "regression",
                       xlabel="Gini importance")
imp.round(4)
"""),
        code(show()),
        team("Feature importance vs linear coefficients",
             "Do the Random Forest importances rank the features the same way the linear coefficients did?",
             "Where they disagree, which do you trust for this dataset, and why?"),

        md("## 9. Review 1 - regression conclusion"),
        team("Regression track conclusion",
             "Which model do you nominate as final, and on what evidence (test metric, CV stability, simplicity, training cost)?",
             "What is the practical meaning of the RMSE in Performance-Index points?",
             "What is the main limitation of this dataset, given that it is synthetic?",
             "What would you do next if you had another week?"),
    ]
    return cells


# ==========================================================================
def classification_nb():
    spec = config.CLASSIFICATION
    cells = [
        md(HEADER.format(title="Classification Track - Machine Predictive Maintenance (Part A)",
                         review="Review 1 - Part A (the five Review 1 algorithms)",
                         dataset=f"{spec['name']} - `{spec['filename']}`",
                         url=spec["url"])),
        md("## 0. Setup"),
        code(SETUP),
        code(f"""
TARGET = "Target"
CLS = ["No Failure", "Failure"]
NUM = {spec["numeric_features"]!r}
CAT = {spec["categorical_features"]!r}
print("target:", TARGET, "| classes:", CLS)
"""),

        md("""
## 1. Data loading, audit and leakage exclusions  *(Rubric A1)*

**Target leakage, handled explicitly.** `Failure Type` is **not** a predictor: it
records the same event as `Target`. A model given it would score near-perfectly
and learn nothing. `UDI` and `Product ID` are row identifiers. All three are
dropped.
"""),
        code("""
raw = dl.load_raw("classification")
summary = dl.audit_summary(raw, "classification")
for k, v in summary.items():
    print(f"  {k}: {v}")
"""),
        code("""
# The evidence for excluding Failure Type: it is determined by Target.
pd.crosstab(raw["Failure Type"], raw["Target"])
"""),
        code("""dl.audit(raw, "classification", "raw")"""),
        code("""
df = dl.prepare("classification", raw)
print("\\nprepared columns:", list(df.columns))
df.head()
"""),

        md("""
## 2. Exploratory data analysis  *(Rubric A2, A3)*

The target is binary and heavily imbalanced, so a plain feature-vs-target
scatter would collapse onto two horizontal lines. We add jitter and a companion
box plot.
"""),
        code("""
p = pl.distribution_grid(df[NUM + CAT],
                         "Predictive Maintenance - input feature distributions",
                         "nb_clf_feature_distributions", "classification")
"""),
        code(show()),
        team("Feature distributions",
             "Which features look approximately normal, and which are skewed or bounded?",
             "Air and process temperature: what does their shape suggest about how this synthetic data was generated?",
             "Does the Type category split the machines evenly?"),
        code("""
p = pl.correlation_heatmap(df, "Predictive Maintenance - correlation heatmap (numeric)",
                           "nb_clf_correlation_heatmap", "classification")
"""),
        code(show()),
        code("""df.select_dtypes("number").corr().round(3)"""),
        team("Correlation structure",
             "Two pairs of features are strongly related. Which pairs, and why physically?",
             "Does any single feature correlate strongly with Target on its own? What does that imply about whether failure is a single-variable phenomenon?"),
        code("""
p = pl.target_distribution(df[TARGET], "Target distribution - machine failure",
                           "nb_clf_target_distribution", "classification",
                           discrete=True, class_names={0: CLS[0], 1: CLS[1]})
"""),
        code(show()),
        team("Class imbalance",
             "What fraction of rows are failures?",
             "What accuracy would a model get by predicting 'No Failure' every single time?",
             "Which metric should therefore lead your comparison, and why not accuracy?"),
        code("""
p = pl.feature_target_scatter(df, ["Torque [Nm]", "Rotational speed [rpm]"], TARGET,
                              "nb_clf_feature_target_scatter", "classification",
                              jitter=0.08,
                              title="Feature-target relationships (target jittered for readability)")
"""),
        code(show()),
        code("""
p = pl.grouped_box(df, ["Torque [Nm]", "Rotational speed [rpm]", "Tool wear [min]"],
                   TARGET, "nb_clf_feature_by_class", "classification",
                   class_names={0: CLS[0], 1: CLS[1]})
"""),
        code(show()),
        code("""
print("mean of each feature, by class:")
df.groupby(TARGET).mean(numeric_only=True).round(3).T
"""),
        team("Feature-target relationships",
             "In which region of torque and rotational speed do failures concentrate?",
             "Do the box plots separate the two classes on any single feature, or is failure an interaction between features?",
             "What does that suggest about linear versus tree-based models here?"),

        md("""
## 3. Split, baseline and feature engineering  *(Rubric B1, B2, B3)*

**Stratified** 80:20. With only 3.4% positives, an unstratified split could give
train and test noticeably different failure rates and make the comparison unfair.

As in the regression notebook, all preprocessing lives inside each Pipeline, so
scalers and encoders are fitted on training data only and refitted per CV fold.
"""),
        code("""
X_train, X_test, y_train, y_test = pp.split_supervised(df, TARGET, stratify=True)
print(f"\\ntrain: {X_train.shape}   test: {X_test.shape}")
print("train failure rate:", round(y_train.mean() * 100, 3), "%")
print("test  failure rate:", round(y_test.mean() * 100, 3), "%")
"""),
        code("""
baseline = ev.majority_class_baseline(y_train, y_test)
baseline
"""),
        md("""
> **Read the baseline before any model score.** Every model below must be judged
> against this number, not against zero.
"""),
        code("""fe.status("classification")"""),
        team("Engineered feature (Q-CF1) - required for rubric B3",
             "Which physically meaningful quantity could you derive from the existing columns?",
             "State your justification BEFORE measuring it.",
             "After registering it and re-running, did failure recall improve? Report honestly."),

        md("""
## 4. The five Part A algorithms  *(Rubric D1, D2)*

**Metric conventions, stated once and used everywhere:**

* Precision and Recall are **binary**, with **failure (`Target == 1`) as the positive class**
* F1 is **weighted**, as the rubric mandates
* ROC-AUC uses `predict_proba` / `decision_function` - **never** hard predictions

**What each metric means here:**

| Metric | Plain meaning |
|---|---|
| Accuracy | share of all 2,000 test machines classified correctly |
| Precision (failure) | of the machines we called failures, how many really failed |
| Recall (failure) | of the machines that really failed, how many we caught |
| F1 (weighted) | harmonic mean of precision and recall, averaged over classes by size |
| ROC-AUC | probability the model scores a random failure above a random healthy machine |
"""),
        code("""
from sklearn.metrics import confusion_matrix

results = {}     # model name -> metrics
fitted  = {}     # model name -> fitted Pipeline
cms     = {}     # model name -> confusion matrix

def record(name, model):
    \"\"\"Evaluate on the held-out test set, print a readable report, store it.\"\"\"
    y_pred = model.predict(X_test)
    y_score, how = ev._scores_for_auc(model, X_test)
    m = ev.classification_metrics(y_test, y_pred, y_score)
    m["AUC_source"] = how
    results[name] = m
    fitted[name] = model
    cm = confusion_matrix(y_test, y_pred)
    cms[name] = cm
    tn, fp, fn, tp = cm.ravel()
    print(f"{name}")
    print(f"  Accuracy          : {m['Accuracy']:.4f}   (baseline {baseline['baseline_accuracy']:.4f})")
    print(f"  Precision(failure): {m['Precision_failure']:.4f}")
    print(f"  Recall(failure)   : {m['Recall_failure']:.4f}")
    print(f"  F1 (weighted)     : {m['F1_weighted']:.4f}")
    print(f"  ROC-AUC           : {m['ROC_AUC']:.4f}   (from {how})")
    print(f"  confusion: correct-safe={tn}  false-alarms={fp}  MISSED={fn}  caught={tp}")
    return m
"""),
    ]

    for i, m in enumerate(CLASSIFICATION[:5], start=1):
        cells.append(md(algo_markdown(m, f"4.{i}")))
        cells.append(code(f"""{strip_pipeline_import(m["imports"])}

model = Pipeline([
    ("prep", pp.make_preprocessor(NUM, CAT, scale={m['scale']})),
    ("model", {m['estimator']}),
])

model.fit(X_train, y_train)
record("{m['title'].split('  [')[0]}", model)
"""))
        if m["extra"].strip():
            cells.append(code(m["extra"].strip()))
            if "decision_tree_figure" in m["extra"]:
                cells.append(code(show()))

    cells += [
        md("""
## 5. Part A comparison  *(Rubric D2 - Review 1 deliverable)*

One table, all five Part A algorithms, same dataset and same held-out split.
"""),
        code("""
cols = ["Model", "Accuracy", "Precision_failure", "Recall_failure",
        "F1_weighted", "ROC_AUC", "AUC_source"]
table = (pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
         .sort_values("F1_weighted", ascending=False).reset_index(drop=True))[cols]
table.index = table.index + 1
table.index.name = "Rank"
print(f"{len(table)} models | majority-class baseline: "
      f"accuracy {baseline['baseline_accuracy']:.4f}, "
      f"weighted F1 {baseline['baseline_f1_weighted']:.4f}\\n")
table
"""),
        code("""
p = pl.model_comparison_bar(table, "F1_weighted",
                            "Part A classifiers ranked by weighted F1",
                            "nb_clf_model_comparison", "classification")
"""),
        code(show()),
        code("""
p = pl.confusion_grid(cms, CLS, "Part A confusion matrices (held-out test set)",
                      "nb_clf_confusion_partA", "classification")
"""),
        code(show()),
        code("""
p = pl.roc_curves(fitted, X_test, y_test,
                  "ROC curves (probability / decision scores, not hard labels)",
                  "nb_clf_roc_curves", "classification")
"""),
        code(show()),
        team("Part A comparison (Review 1 deliverable)",
             "Rank the five models on weighted F1, then on failure recall. Does the ranking change?",
             "Look at the confusion matrices: which models achieve high accuracy mainly by rarely predicting failure at all?",
             "In a maintenance setting, which error costs more - a missed failure or a false alarm? Which model does your answer favour?",
             "Accuracy spans a narrow range while failure recall spans a wide one. What does that tell you?"),

        md("""\n## 6. Cross-validation and tuning\n\nLeading models are nominated on **training-side** cross-validation; the\nheld-out test set plays no part in selection.\n"""),
        code("""
cv_all = ev.cv_scores(fitted, X_train, y_train, scoring="f1_weighted")
cv_all.drop(columns="fold_scores").sort_values("CV_mean", ascending=False)
"""),
        code("""
from src import classification_models as cm

leaders = cv_all.sort_values("CV_mean", ascending=False)["Model"].head(2).tolist()
print("leaders selected on TRAINING cross-validation:", leaders)

rows, tuned = {}, {}
for name in leaders:
    if name not in cm.PARAM_GRIDS:
        continue
    before_cv = float(cv_all.loc[cv_all["Model"] == name, "CV_mean"].iloc[0])
    before = table[table["Model"] == name].iloc[0]
    gs = cm.tune(name, fitted[name], X_train, y_train)
    score, _ = ev._scores_for_auc(gs.best_estimator_, X_test)
    after = ev.classification_metrics(y_test, gs.best_estimator_.predict(X_test), score)
    rows[name] = {
        "best_params": gs.best_params_,
        "CV_F1w_before": round(before_cv, 5),
        "CV_F1w_after": round(gs.best_score_, 5),
        "HeldOut_F1w_before": round(float(before["F1_weighted"]), 5),
        "HeldOut_F1w_after": round(after["F1_weighted"], 5),
        "Recall_fail_before": round(float(before["Recall_failure"]), 5),
        "Recall_fail_after": round(after["Recall_failure"], 5),
        "improved_on_CV": bool(gs.best_score_ > before_cv),
    }
    tuned[name] = gs.best_estimator_
pd.DataFrame(rows).T
"""),
        team("Final model selection",
             "Which model do you select as final, and which metric drove the decision?",
             "Did tuning improve at least one metric? Quote the before/after numbers - and if it did not, say so.",
             "Does your chosen model trade failure recall for accuracy? Is that trade acceptable for the use case?"),

        md("## 7. Classification (Part A) conclusion"),
        team("Classification conclusion",
             "Summarise Part A: problem -> data -> method -> result -> conclusion.",
             "What is the single most important caveat a reader should keep in mind (imbalance? synthetic data? threshold choice?)",
             "If this were deployed on a real production line, what would you monitor?"),
    ]
    return cells


# ==========================================================================
def write(name: str, cells: list) -> Path:
    cells = [c for c in cells if c is not None]     # drop retired team() markers
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    NB_DIR.mkdir(parents=True, exist_ok=True)
    path = NB_DIR / f"{name}.ipynb"
    nbf.write(nb, path)
    n_code = sum(1 for c in cells if c.cell_type == "code")
    n_team = sum(1 for c in cells
                 if c.cell_type == "markdown" and "TEAM TO COMPLETE" in c.source)
    print(f"[notebook] {path.relative_to(config.PROJECT_ROOT)} "
          f"({len(cells)} cells: {n_code} code, {n_team} team cells)")
    return path


def main() -> int:
    write("regression", regression_nb())
    write("classification", classification_nb())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
