"""The ten regression algorithms required by PDF section 3.1.

Each entry is a full Pipeline (preprocess -> estimator), so preprocessing is
refitted inside every cross-validation fold and the saved artifact can predict
straight from raw-shaped input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from . import config
from .feature_engineering import FeatureAdder
from .preprocessing import make_preprocessor

RS = config.RANDOM_STATE


def _pipe(numeric, categorical, estimator, scale=True, poly_degree=None):
    steps = [("features", FeatureAdder(track="regression")),
             ("prep", make_preprocessor(numeric, categorical, scale=scale))]
    if poly_degree:
        steps.append(("poly", PolynomialFeatures(degree=poly_degree,
                                                 include_bias=False)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def build_models(numeric: list[str], categorical: list[str],
                 poly_degree: int = 2) -> dict[str, Pipeline]:
    """All ten regressors.

    Scaling notes (PDF algorithm notes):
      * linear family, SVR and KNN are distance/penalty sensitive -> scaled
      * tree and ensemble models are scale-invariant -> unscaled, which keeps
        feature importances tied to the original units
    """
    return {
        "1. Linear Regression":
            _pipe(numeric, categorical, LinearRegression()),
        "2. Ridge Regression":
            _pipe(numeric, categorical, Ridge(alpha=1.0, random_state=RS)),
        "3. Lasso Regression":
            _pipe(numeric, categorical, Lasso(alpha=0.01, random_state=RS, max_iter=10000)),
        "4. ElasticNet Regression":
            _pipe(numeric, categorical, ElasticNet(alpha=0.01, l1_ratio=0.5,
                                                   random_state=RS, max_iter=10000)),
        f"5. Polynomial (deg {poly_degree}) + Linear":
            _pipe(numeric, categorical, LinearRegression(), poly_degree=poly_degree),
        "6. Decision Tree Regressor":
            _pipe(numeric, categorical, DecisionTreeRegressor(max_depth=8, random_state=RS),
                  scale=False),
        "7. Random Forest Regressor":
            _pipe(numeric, categorical,
                  RandomForestRegressor(n_estimators=300, random_state=RS, n_jobs=-1),
                  scale=False),
        "8. Gradient Boosting Regressor":
            _pipe(numeric, categorical,
                  GradientBoostingRegressor(learning_rate=0.1, n_estimators=200,
                                            random_state=RS),
                  scale=False),
        "9. Support Vector Regressor":
            _pipe(numeric, categorical, SVR(C=1.0, kernel="rbf")),
        "10. K-Nearest Neighbors Regressor":
            _pipe(numeric, categorical, KNeighborsRegressor(n_neighbors=5)),
    }


# --------------------------------------------------------------------------
# Algorithm-specific displays required by the PDF notes column
# --------------------------------------------------------------------------
def coefficient_table(fitted_pipe: Pipeline) -> pd.DataFrame:
    """Coefficients of a fitted linear-family pipeline, with readable names."""
    prep = fitted_pipe.named_steps["prep"]
    names = list(prep.get_feature_names_out())
    if "poly" in fitted_pipe.named_steps:
        names = list(fitted_pipe.named_steps["poly"].get_feature_names_out(names))
    model = fitted_pipe.named_steps["model"]
    df = pd.DataFrame({"feature": names, "coefficient": model.coef_.ravel()})
    df["abs"] = df["coefficient"].abs()
    df = df.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)
    df.attrs["intercept"] = float(np.ravel(model.intercept_)[0])
    return df


def sparsity_report(fitted_lasso: Pipeline, tol: float = 1e-8) -> dict:
    """Lasso feature sparsity (PDF note for algorithm 3)."""
    coefs = fitted_lasso.named_steps["model"].coef_.ravel()
    names = list(fitted_lasso.named_steps["prep"].get_feature_names_out())
    zeroed = [n for n, c in zip(names, coefs) if abs(c) <= tol]
    return {"n_features": len(coefs), "n_zero_coefficients": len(zeroed),
            "zeroed_features": zeroed,
            "pct_zero": round(100 * len(zeroed) / max(len(coefs), 1), 2)}


def polynomial_degree_comparison(numeric, categorical, X_train, y_train,
                                 X_test, y_test, degrees=(1, 2, 3)) -> pd.DataFrame:
    """PDF note for algorithm 5: compare polynomial degrees."""
    from .evaluation import regression_metrics
    rows = []
    for d in degrees:
        p = _pipe(numeric, categorical, LinearRegression(), poly_degree=d)
        p.fit(X_train, y_train)
        m_tr = regression_metrics(y_train, p.predict(X_train))
        m_te = regression_metrics(y_test, p.predict(X_test))
        n_terms = p.named_steps["poly"].n_output_features_
        rows.append({"degree": d, "n_expanded_terms": n_terms,
                     "train_R2": m_tr["R2"], "test_R2": m_te["R2"],
                     "test_RMSE": m_te["RMSE"], "test_MAE": m_te["MAE"]})
    return pd.DataFrame(rows)


def tree_importances(fitted_pipe: Pipeline) -> pd.Series:
    prep = fitted_pipe.named_steps["prep"]
    names = list(prep.get_feature_names_out())
    return pd.Series(fitted_pipe.named_steps["model"].feature_importances_,
                     index=names).sort_values(ascending=False)


# --------------------------------------------------------------------------
# Tuning grids (PDF: tune alpha, l1_ratio, max_depth, n_estimators,
# learning_rate, C/kernel, k)
# --------------------------------------------------------------------------
PARAM_GRIDS = {
    "2. Ridge Regression": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "3. Lasso Regression": {"model__alpha": [0.001, 0.01, 0.1, 1.0]},
    "4. ElasticNet Regression": {"model__alpha": [0.001, 0.01, 0.1, 1.0],
                                 "model__l1_ratio": [0.1, 0.5, 0.9]},
    "6. Decision Tree Regressor": {"model__max_depth": [3, 5, 8, 12, None],
                                   "model__min_samples_leaf": [1, 5, 20]},
    "7. Random Forest Regressor": {"model__n_estimators": [100, 300, 500],
                                   "model__max_depth": [None, 10, 20]},
    "8. Gradient Boosting Regressor": {"model__learning_rate": [0.03, 0.1, 0.2],
                                       "model__n_estimators": [100, 200, 400]},
    "9. Support Vector Regressor": {"model__C": [0.1, 1.0, 10.0],
                                    "model__kernel": ["rbf", "linear"]},
    "10. K-Nearest Neighbors Regressor": {"model__n_neighbors": [3, 5, 9, 15, 25],
                                          "model__weights": ["uniform", "distance"]},
}


def tune(model_name: str, pipeline: Pipeline, X_train, y_train,
         cv: int = config.CV_FOLDS, scoring: str = "r2") -> GridSearchCV:
    grid = PARAM_GRIDS[model_name]
    gs = GridSearchCV(pipeline, grid, cv=cv, scoring=scoring, n_jobs=-1,
                      return_train_score=True)
    gs.fit(X_train, y_train)
    print(f"  [tuned] {model_name}: best {scoring}={gs.best_score_:.4f} "
          f"params={gs.best_params_}")
    return gs
