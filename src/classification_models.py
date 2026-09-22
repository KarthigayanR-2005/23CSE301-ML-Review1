"""The five PART A classification algorithms required by PDF section 3.2.

Part A (Review 1): Logistic Regression, KNN, Gaussian NB, Decision Tree, SVC

Part B (Random Forest, AdaBoost, Gradient Boosting, Bagging, MLP) belongs to
Review 2 and is deliberately absent from this repository. See README.md "Scope".

Every model is a Pipeline over the SAME prepared dataset and the SAME held-out
split, which is what makes the Part A comparison table legitimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config
from .feature_engineering import FeatureAdder
from .preprocessing import make_preprocessor

RS = config.RANDOM_STATE


def _pipe(numeric, categorical, estimator, scale=True):
    return Pipeline([
        ("features", FeatureAdder(track="classification")),
        ("prep", make_preprocessor(numeric, categorical, scale=scale)),
        ("model", estimator),
    ])


def build_part_a(numeric, categorical) -> dict[str, Pipeline]:
    """The five Review 1 (Part A) models.

    GaussianNB is required explicitly by the PDF - do not substitute
    MultinomialNB or BernoulliNB.
    """
    return {
        "A1. Logistic Regression":
            _pipe(numeric, categorical,
                  LogisticRegression(max_iter=2000, random_state=RS)),
        "A2. K-Nearest Neighbors":
            _pipe(numeric, categorical, KNeighborsClassifier(n_neighbors=5)),
        "A3. Gaussian Naive Bayes":
            _pipe(numeric, categorical, GaussianNB()),
        "A4. Decision Tree Classifier":
            _pipe(numeric, categorical,
                  DecisionTreeClassifier(max_depth=6, random_state=RS), scale=False),
        "A5. Support Vector Classifier":
            _pipe(numeric, categorical,
                  SVC(C=1.0, kernel="rbf", probability=True, random_state=RS)),
    }


# --------------------------------------------------------------------------
# Algorithm-specific displays
# --------------------------------------------------------------------------
def odds_table(fitted_logreg: Pipeline) -> pd.DataFrame:
    """Coefficients and odds ratios (PDF note for algorithm 1).

    Because the numeric block is standardised, each coefficient is the log-odds
    change per ONE STANDARD DEVIATION of that feature.
    """
    names = list(fitted_logreg.named_steps["prep"].get_feature_names_out())
    coefs = fitted_logreg.named_steps["model"].coef_.ravel()
    df = pd.DataFrame({
        "feature": names,
        "coefficient_log_odds": coefs,
        "odds_ratio": np.exp(coefs),
    })
    df["abs"] = df["coefficient_log_odds"].abs()
    df = df.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)
    df.attrs["intercept"] = float(np.ravel(fitted_logreg.named_steps["model"].intercept_)[0])
    df.attrs["scale_note"] = "numeric features are standardised; units are per 1 SD"
    return df


def tree_importances(fitted_pipe: Pipeline) -> pd.Series:
    names = list(fitted_pipe.named_steps["prep"].get_feature_names_out())
    return pd.Series(fitted_pipe.named_steps["model"].feature_importances_,
                     index=names).sort_values(ascending=False)


def knn_distance_metric_comparison(numeric, categorical, X_train, y_train,
                                   X_test, y_test,
                                   metrics=("euclidean", "manhattan", "chebyshev"),
                                   k: int = 5) -> pd.DataFrame:
    """PDF note for algorithm 2: discuss distance metrics."""
    from .evaluation import classification_metrics, _scores_for_auc
    rows = []
    for met in metrics:
        p = _pipe(numeric, categorical, KNeighborsClassifier(n_neighbors=k, metric=met))
        p.fit(X_train, y_train)
        score, _ = _scores_for_auc(p, X_test)
        m = classification_metrics(y_test, p.predict(X_test), score)
        m["distance_metric"] = met
        rows.append(m)
    cols = ["distance_metric", "Accuracy", "Precision_failure", "Recall_failure",
            "F1_weighted", "ROC_AUC"]
    return pd.DataFrame(rows)[cols]


PARAM_GRIDS = {
    "A1. Logistic Regression": {"model__C": [0.01, 0.1, 1.0, 10.0],
                                "model__class_weight": [None, "balanced"]},
    "A2. K-Nearest Neighbors": {"model__n_neighbors": [3, 5, 9, 15, 25],
                                "model__weights": ["uniform", "distance"]},
    "A4. Decision Tree Classifier": {"model__max_depth": [3, 5, 8, 12, None],
                                     "model__min_samples_leaf": [1, 5, 20],
                                     "model__class_weight": [None, "balanced"]},
    "A5. Support Vector Classifier": {"model__C": [0.1, 1.0, 10.0],
                                      "model__kernel": ["rbf", "linear"]},
    # NOTE: Gaussian Naive Bayes has no meaningful hyperparameters to tune.
}


def tune(model_name: str, pipeline: Pipeline, X_train, y_train,
         cv: int = config.CV_FOLDS, scoring: str = "f1_weighted") -> GridSearchCV:
    gs = GridSearchCV(pipeline, PARAM_GRIDS[model_name], cv=cv, scoring=scoring,
                      n_jobs=-1, return_train_score=True)
    gs.fit(X_train, y_train)
    print(f"  [tuned] {model_name}: best {scoring}={gs.best_score_:.4f} "
          f"params={gs.best_params_}")
    return gs


def build_all(numeric, categorical) -> dict[str, Pipeline]:
    """Every classifier in THIS repository - i.e. Part A only (Review 1)."""
    return build_part_a(numeric, categorical)
