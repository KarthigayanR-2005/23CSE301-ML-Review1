"""Streamlit interface for the REVIEW 1 capstone deliverable.

Loads the SAVED pipelines from models/ - preprocessing is not recreated here,
so what the app does to an input is exactly what was done during training.

Run locally:
    pip install -r requirements.txt
    streamlit run app/streamlit_app.py

NOTE ON DEPLOYMENT: this app has NOT been deployed to any public URL. The
second bonus mark requires a live, verified URL. See README.md § Bonus.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src import config

st.set_page_config(page_title="23CSE301 ML Capstone (Review 1)", page_icon="🎓", layout="wide")


@st.cache_resource
def load(name: str):
    p = config.MODELS_DIR / name
    return joblib.load(p) if p.exists() else None


reg_model = load("regression_best_pipeline.joblib")
reg_meta = load("regression_best_metadata.joblib")
clf_model = load("classification_best_pipeline.joblib")
clf_meta = load("classification_best_metadata.joblib")

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
    if reg_model is None:
        st.error("No saved regression pipeline. Run `python scripts/run_all.py "
                 "--track regression --mode full` first.")
    else:
        st.caption(f"Model: **{reg_meta['model_name']}**")
        c1, c2 = st.columns(2)
        with c1:
            hours = st.slider("Hours Studied", 1, 9, 5,
                              help="Training data covers 1-9 hours.")
            prev = st.slider("Previous Scores", 40, 99, 70)
            sleep = st.slider("Sleep Hours", 4, 9, 7)
        with c2:
            papers = st.slider("Sample Question Papers Practiced", 0, 9, 4)
            extra = st.radio("Extracurricular Activities", ["Yes", "No"], horizontal=True)

        if st.button("Predict performance index", type="primary"):
            X = pd.DataFrame([{
                "Hours Studied": hours, "Previous Scores": prev,
                "Sleep Hours": sleep, "Sample Question Papers Practiced": papers,
                "Extracurricular Activities": extra,
            }])
            pred = float(reg_model.predict(X)[0])
            pred_clipped = float(np.clip(pred, 10, 100))
            st.metric("Predicted Performance Index", f"{pred_clipped:.1f}")
            if abs(pred - pred_clipped) > 0.05:
                st.info(f"Raw model output was {pred:.1f}; clipped to the 10-100 range "
                        "observed in training. Regression models are not bounded.")
            rmse = reg_meta["sklearn_metrics"]["RMSE"]
            st.caption(
                f"**How to read this.** Typical error on unseen data was "
                f"RMSE ≈ {rmse:.2f} index points, so treat this as roughly "
                f"{pred_clipped:.0f} ± {rmse:.0f}. The single strongest driver in "
                f"the data is Previous Scores, so a prediction mostly reflects "
                f"where a student already stood.")

# ------------------------------------------------------------ classification
with tab2:
    st.header("Machine failure risk — Part A model")
    if clf_model is None:
        st.error("No saved classification pipeline. Run `python scripts/run_all.py "
                 "--track classification --mode full` first.")
    else:
        st.caption(f"Model: **{clf_meta['model_name']}**")
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
            st.warning("Process temperature is normally above air temperature in this "
                       "dataset. The prediction below is extrapolating.")

        if st.button("Assess failure risk", type="primary"):
            X = pd.DataFrame([{
                "Air temperature [K]": air, "Process temperature [K]": proc,
                "Rotational speed [rpm]": rpm, "Torque [Nm]": torque,
                "Tool wear [min]": wear, "Type": mtype,
            }])
            pred = int(clf_model.predict(X)[0])
            proba = (float(clf_model.predict_proba(X)[0, 1])
                     if hasattr(clf_model, "predict_proba") else None)
            label = clf_meta["classes"][pred]
            (st.error if pred == 1 else st.success)(f"Prediction: **{label}**")
            if proba is not None:
                st.progress(min(proba, 1.0))
                st.metric("Estimated failure probability", f"{proba*100:.1f}%")
            st.caption(
                "**How to read this.** Only about 3.4% of the training rows were "
                "failures, so the model sees failure as rare by default and a "
                "'No Failure' output is the unsurprising answer. Pay attention to "
                "the probability rather than the label: a 20% probability is far "
                "above the base rate even though the label still reads 'No Failure'. "
                "This is a single Decision Tree, so its probability is the class "
                "proportion in whichever leaf your inputs land in - expect a small "
                "set of repeated values rather than a smooth scale.")

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

**Deployment status:** this app runs locally only. It has **not** been deployed
to a public URL, so no public-deployment claim is made.

**AI assistance:** code scaffolding was AI-generated and disclosed in
`README.md`. Analysis, interpretation and feature-engineering decisions are
team-authored.
""")
    st.subheader("Model artifacts loaded")
    st.json({
        "regression_pipeline": reg_model is not None,
        "classification_pipeline": clf_model is not None,
        "regression_model": reg_meta["model_name"] if reg_meta else None,
        "classification_model": clf_meta["model_name"] if clf_meta else None,
    })
