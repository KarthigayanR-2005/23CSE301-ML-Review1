"""Streamlit interface for the REVIEW 1 capstone deliverable.

Loads the SAVED pipelines from models/ - preprocessing is not recreated here,
so what the app does to an input is exactly what was done during training.

Every model from both comparison tables is available from a dropdown, listed in
the same rank order as the report: regression by test R2, classification by
weighted F1. That makes it possible to watch, for example, Logistic Regression
miss a failure that the Decision Tree catches on identical inputs.

Run locally:
    pip install -r requirements.txt
    streamlit run app/streamlit_app.py

NOTE ON DEPLOYMENT: this app has NOT been deployed to any public URL. No
public-deployment claim is made. See README.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src import config

st.set_page_config(page_title="23CSE301 ML Capstone (Review 1)",
                   page_icon="🎓", layout="wide")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@st.cache_data
def load_index(track: str) -> list[dict]:
    """The ranked model list written by scripts/run_all.py."""
    p = config.MODELS_DIR / f"{track}_index.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())


@st.cache_resource
def load_model(relpath: str):
    """One saved pipeline. Cached, so switching models stays snappy."""
    return joblib.load(config.MODELS_DIR / relpath)


@st.cache_resource
def load_meta(name: str):
    p = config.MODELS_DIR / name
    return joblib.load(p) if p.exists() else None


reg_index = load_index("regression")
clf_index = load_index("classification")
reg_meta = load_meta("regression_best_metadata.joblib")
clf_meta = load_meta("classification_best_metadata.joblib")


def picker(track: str, index: list[dict], metric_label: str, key: str) -> dict | None:
    """Dropdown listing every model in report rank order. Returns the entry."""
    if not index:
        st.error(
            f"No saved {track} models found. Run "
            "`python scripts/run_all.py --mode full` first.")
        return None

    labels = [
        f"{e['rank']}. {e['model']}  —  {metric_label} {e['rank_value']:.4f}"
        for e in index
    ]
    choice = st.selectbox(
        f"Model  ({len(index)} available, ranked by {metric_label} on the "
        "held-out test set)",
        options=range(len(index)),
        format_func=lambda i: labels[i],
        key=key,
    )
    entry = index[choice]
    if entry["rank"] == 1:
        st.caption("🥇 Top-ranked model for this track.")
    if entry.get("also_tuned"):
        st.caption(
            "This model was also hyperparameter-tuned. The pipeline loaded here "
            "is the **baseline**, so its score matches the comparison table; "
            "the tuned figures are in `results/tables/*_tuning.csv`.")
    return entry


st.title("23CSE301 Machine Learning — Capstone (Review 1)")
st.caption("Predictions come from the saved training pipelines. "
           "Preprocessing is loaded, not re-implemented.")

st.warning(
    "**Both supervised models were trained on SYNTHETIC datasets.** The student "
    "performance data and the predictive-maintenance data are both artificially "
    "generated. Outputs are a coursework demonstration and must not be used to "
    "judge a real student or a real machine.", icon="⚠️")

tab1, tab2, tab3 = st.tabs(
    ["📈 Regression", "⚙️ Classification (Part A)", "ℹ️ About"])

# ---------------------------------------------------------------- regression
with tab1:
    st.header("Student Performance Index")
    entry = picker("regression", reg_index, "R²", "reg_pick")

    if entry:
        c1, c2 = st.columns(2)
        with c1:
            hours = st.slider("Hours Studied", 1, 9, 5,
                              help="Training data covers 1-9 hours.")
            prev = st.slider("Previous Scores", 40, 99, 70)
            sleep = st.slider("Sleep Hours", 4, 9, 7)
        with c2:
            papers = st.slider("Sample Question Papers Practiced", 0, 9, 4)
            extra = st.radio("Extracurricular Activities", ["Yes", "No"],
                             horizontal=True)

        if st.button("Predict performance index", type="primary"):
            X = pd.DataFrame([{
                "Hours Studied": hours, "Previous Scores": prev,
                "Sleep Hours": sleep,
                "Sample Question Papers Practiced": papers,
                "Extracurricular Activities": extra,
            }])
            model = load_model(entry["file"])
            pred = float(model.predict(X)[0])
            pred_clipped = float(np.clip(pred, 10, 100))

            st.metric("Predicted Performance Index", f"{pred_clipped:.1f}")
            if abs(pred - pred_clipped) > 0.05:
                st.info(f"Raw model output was {pred:.1f}; clipped to the 10-100 "
                        "range observed in training. Regression models are not "
                        "bounded.")

            rmse = entry.get("RMSE")
            if rmse:
                st.caption(
                    f"**How to read this.** This model's typical error on unseen "
                    f"data was RMSE ≈ {rmse:.2f} index points, so treat the "
                    f"answer as roughly {pred_clipped:.0f} ± {rmse:.0f}. The "
                    f"single strongest driver in the data is Previous Scores, so "
                    f"a prediction mostly reflects where a student already stood.")

            with st.expander("Compare every model on these same inputs"):
                rows = []
                for e in reg_index:
                    v = float(load_model(e["file"]).predict(X)[0])
                    rows.append({
                        "Rank": e["rank"], "Model": e["model"],
                        "Prediction": round(float(np.clip(v, 10, 100)), 1),
                        "Test R²": round(e["rank_value"], 6),
                        "RMSE": round(e.get("RMSE", float("nan")), 4),
                    })
                st.dataframe(pd.DataFrame(rows).set_index("Rank"),
                             width="stretch")
                st.caption("All ten agree closely here — the spread across the "
                           "whole table is about 1.2% R². That is characteristic "
                           "of this synthetic dataset, not of regression in "
                           "general.")

# ------------------------------------------------------------ classification
with tab2:
    st.header("Machine failure risk — Part A models")
    entry = picker("classification", clf_index, "weighted F1", "clf_pick")

    if entry:
        c1, c2 = st.columns(2)
        with c1:
            mtype = st.selectbox("Type (product quality)", ["L", "M", "H"], index=1)
            air = st.number_input("Air temperature [K]", 295.0, 305.0, 300.0, 0.1)
            proc = st.number_input("Process temperature [K]", 305.0, 315.0, 310.0, 0.1)
        with c2:
            rpm = st.number_input("Rotational speed [rpm]", 1100, 2900, 1500, 10)
            torque = st.number_input("Torque [Nm]", 3.0, 77.0, 40.0, 0.1)
            wear = st.number_input("Tool wear [min]", 0, 260, 100, 1)

        if proc <= air:
            st.warning("Process temperature is normally above air temperature in "
                       "this dataset. The prediction below is extrapolating.")

        if st.button("Assess failure risk", type="primary"):
            X = pd.DataFrame([{
                "Air temperature [K]": air, "Process temperature [K]": proc,
                "Rotational speed [rpm]": rpm, "Torque [Nm]": torque,
                "Tool wear [min]": wear, "Type": mtype,
            }])
            model = load_model(entry["file"])
            classes = (clf_meta["classes"] if clf_meta
                       else {0: "No Failure", 1: "Failure"})
            pred = int(model.predict(X)[0])
            proba = (float(model.predict_proba(X)[0, 1])
                     if hasattr(model, "predict_proba") else None)

            (st.error if pred == 1 else st.success)(
                f"Prediction: **{classes[pred]}**")
            if proba is not None:
                st.progress(min(proba, 1.0))
                st.metric("Estimated failure probability", f"{proba*100:.1f}%")

            rec = entry.get("Recall_failure")
            st.caption(
                "**How to read this.** Only about 3.4% of the training rows were "
                "failures, so every model treats failure as rare and a "
                "'No Failure' output is the unsurprising answer. Watch the "
                "probability rather than the label."
                + (f" This model caught **{rec*100:.0f}%** of real failures in "
                   f"testing — so a 'No Failure' from it is weaker evidence than "
                   f"it sounds." if rec is not None else ""))

            with st.expander("Compare every model on these same inputs"):
                rows = []
                for e in clf_index:
                    m = load_model(e["file"])
                    p = int(m.predict(X)[0])
                    pr = (float(m.predict_proba(X)[0, 1])
                          if hasattr(m, "predict_proba") else float("nan"))
                    rows.append({
                        "Rank": e["rank"], "Model": e["model"],
                        "Prediction": classes[p],
                        "P(failure)": f"{pr*100:.1f}%",
                        "Weighted F1": round(e["rank_value"], 4),
                        "Recall (fail)": round(e.get("Recall_failure",
                                                     float("nan")), 4),
                    })
                st.dataframe(pd.DataFrame(rows).set_index("Rank"),
                             width="stretch")
                st.caption("Try torque 60, speed 1300, wear 200, Type L: the "
                           "models disagree, and the recall column explains why "
                           "— the weaker ones rarely predict failure at all.")

# -------------------------------------------------------------------- about
with tab3:
    st.header("About this project")
    st.markdown("""
**23CSE301 Machine Learning — Capstone Project (Review 1)**, B.Tech. CSE III Year.

Review 1 scope, evaluated on held-out data that no model was tuned against:

| Track | Dataset | Models |
|---|---|---|
| Regression | Student Performance (synthetic) | 10 |
| Classification — **Part A only** | Machine Predictive Maintenance (synthetic) | 5 |

Classification Part B and the clustering track belong to **Review 2** and are
not part of this repository.

**Model picker.** Every model from both comparison tables is saved and
selectable, ordered exactly as the report ranks them — regression by test R²,
classification by weighted F1. The pipelines loaded are the **baseline** fits,
so each model's dropdown score is the one it earned in
`results/tables/*_comparison.csv`. Tuned variants are reported separately in
`results/tables/*_tuning.csv`.

**Deployment status:** this app runs locally only. It has **not** been deployed
to a public URL, so no public-deployment claim is made.

**AI assistance:** code scaffolding was AI-generated and disclosed in
`README.md`. Analysis, interpretation and feature-engineering decisions are
team-authored.
""")
    st.subheader("Loaded artifacts")
    st.json({
        "regression_models_available": len(reg_index),
        "classification_models_available": len(clf_index),
        "best_regression_model": reg_meta["model_name"] if reg_meta else None,
        "best_classification_model": clf_meta["model_name"] if clf_meta else None,
        "classification_scope": clf_meta.get("part") if clf_meta else None,
    })
