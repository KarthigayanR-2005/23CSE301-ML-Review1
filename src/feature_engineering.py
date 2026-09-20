"""Feature-engineering hook.

=============================================================================
 STATUS: AWAITING TEAM INPUT  -  Review 1 rubric item B3 is NOT satisfied yet.
=============================================================================

The capstone guidelines (section 7.5) require that feature-engineering
decisions are the team's own work. This module therefore ships EMPTY: it
contains the machinery to add an engineered feature to any pipeline, but no
feature and no justification.

Until the team fills in `TEAM_FEATURES` below, every pipeline in this project
runs the *baseline* feature set and `validate_project.py` reports rubric item
B3 as INCOMPLETE.

HOW TO ADD YOUR FEATURE
-----------------------
1. Decide, as a team, on at least one engineered feature per supervised track.
2. Write the transformation as a function taking a DataFrame and returning a
   DataFrame with the new column(s) added.
3. Register it in TEAM_FEATURES with a NON-EMPTY `justification` written by
   your team in your own words.
4. Re-run: `python scripts/run_all.py --full`.

Example of the *shape* of an entry (this is a template, not a suggestion -
do not submit it as-is):

    "regression": EngineeredFeature(
        name="<your feature name>",
        columns_added=["<new column>"],
        func=lambda df: df.assign(**{"<new column>": ...}),
        justification="<your team's reasoning, in your own words>",
        author="<team member name>",
    )
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class EngineeredFeature:
    name: str
    columns_added: list[str]
    func: Callable[[pd.DataFrame], pd.DataFrame]
    justification: str
    author: str = ""
    numeric: bool = True

    def is_complete(self) -> bool:
        return bool(self.name) and bool(self.columns_added) \
            and bool(self.justification.strip()) and self.func is not None


# =========================================================================
# TEAM TO COMPLETE - leave empty until your team has decided and justified.
# =========================================================================
TEAM_FEATURES: dict[str, EngineeredFeature] = {
    # "regression": EngineeredFeature(...),
    # "classification": EngineeredFeature(...),
}


class FeatureAdder(BaseEstimator, TransformerMixin):
    """Applies a registered EngineeredFeature inside a Pipeline.

    Kept inside the pipeline (rather than applied to the whole frame up front)
    so that any future data-dependent feature is computed fold-wise.
    """

    def __init__(self, track: str | None = None):
        self.track = track

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feat = TEAM_FEATURES.get(self.track)
        if feat is None:
            return X
        return feat.func(X.copy())


def status(track: str) -> dict:
    """Machine-readable status used by validate_project.py and the README."""
    feat = TEAM_FEATURES.get(track)
    if feat is None:
        return {"track": track, "state": "AWAITING TEAM INPUT",
                "rubric_B3": "INCOMPLETE",
                "detail": "No engineered feature registered; baseline pipeline executed."}
    if not feat.is_complete():
        return {"track": track, "state": "INCOMPLETE",
                "rubric_B3": "INCOMPLETE",
                "detail": f"Feature '{feat.name}' registered but missing a justification."}
    return {"track": track, "state": "COMPLETE", "rubric_B3": "COMPLETE",
            "detail": f"'{feat.name}' -> {feat.columns_added} (author: {feat.author or 'unnamed'})"}


def extra_numeric_columns(track: str) -> list[str]:
    """Columns the preprocessor must treat as numeric once a feature exists."""
    feat = TEAM_FEATURES.get(track)
    if feat is None or not feat.numeric:
        return []
    return list(feat.columns_added)


def all_status() -> pd.DataFrame:
    return pd.DataFrame([status(t) for t in ("regression", "classification")])
