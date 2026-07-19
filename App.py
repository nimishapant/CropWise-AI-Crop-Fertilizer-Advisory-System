"""
app.py — CropWise: AI-Driven Crop & Fertilizer Advisory System
==================================================================
Streamlit web interface. Run with:

    streamlit run app.py

Structure (matches your report's Part A-E structure):
    Tab 1: Core AI System (Part C)  -> Fuzzy logic fertilizer engine
    Tab 2: ML Baseline (Part D)     -> Random Forest / DT / KNN crop prediction
    Tab 3: Model Comparison (Part B/D) -> side-by-side metrics table + chart

Place your real Kaggle "Crop_recommendation.csv" in this same folder before
running for real report numbers. Without it, the app runs on synthetic
demo data so you can still test the UI end-to-end.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

from fuzzy_fertilizer import diagnose_soil, recommend_dosage
from baseline_models import load_dataset, train_and_evaluate, predict_crop

st.set_page_config(page_title="CropWise — AI Crop & Fertilizer Advisor", layout="wide")

DATA_PATH = "Crop_recommendation.csv" if os.path.exists("Crop_recommendation.csv") else None


@st.cache_data
def get_data():
    return load_dataset(DATA_PATH)


@st.cache_resource
def get_trained_models(_df):
    return train_and_evaluate(_df)


st.title("🌾 CropWise — AI-Driven Agricultural Decision Support")
st.caption(
    "Core system: knowledge-based fuzzy-logic fertilizer advisor. "
    "Baseline comparison: Random Forest / Decision Tree / KNN crop classifiers."
)

if DATA_PATH is None:
    st.warning(
        "No Crop_recommendation.csv found in this folder — running on synthetic "
        "demo data. Add the real Kaggle dataset alongside app.py for genuine results."
    )

df = get_data()
comparison_df, fitted_models, scaler, label_encoder = get_trained_models(df)

tab1, tab2, tab3 = st.tabs([
    "🧠 Core System: Fertilizer Advisor (Fuzzy Logic)",
    "🌱 Baseline: Crop Prediction (ML)",
    "📊 Model Comparison (Part B / D)",
])

# ---------------------------------------------------------------------------
# TAB 1 — CORE AI SYSTEM
# ---------------------------------------------------------------------------
with tab1:
    st.header("Knowledge-Based Fertilizer Recommendation Engine")
    st.markdown(
        "This is the **core AI implementation** for Part C. It uses explicit "
        "fuzzy rules (Low / Medium / High membership) over soil nutrient "
        "readings — not a trained ML model — to reason under uncertainty "
        "about fertilizer need."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        n_val = st.slider("Nitrogen — N (kg/ha)", 0, 140, 40)
    with col2:
        p_val = st.slider("Phosphorus — P (kg/ha)", 0, 145, 40)
    with col3:
        k_val = st.slider("Potassium — K (kg/ha)", 0, 205, 40)

    if st.button("Run Fuzzy Diagnosis", type="primary"):
        diagnosis = diagnose_soil(n_val, p_val, k_val)

        st.subheader("Diagnosis Results")
        result_cols = st.columns(3)
        for i, (nutrient, info) in enumerate(diagnosis.items()):
            dosage = recommend_dosage(info["need_score"], nutrient)
            with result_cols[i]:
                st.metric(
                    label=f"{nutrient} — {info['fertilizer']}",
                    value=f"{info['severity']} need",
                    delta=f"need score {info['need_score']}/100",
                )
                st.write(f"Suggested dosage: **{dosage} kg/ha**")
                st.progress(min(info["need_score"] / 100, 1.0))

        with st.expander("Show fuzzification detail (membership degrees)"):
            for nutrient, info in diagnosis.items():
                st.write(f"**{nutrient} = {info['input_value']}**")
                st.json({k: round(v, 3) for k, v in info["memberships"].items()})

        st.info(
            "This severity + dosage output comes entirely from the rule base and "
            "fuzzy membership functions defined in `fuzzy_fertilizer.py` — no "
            "training data was used to produce this recommendation."
        )

# ---------------------------------------------------------------------------
# TAB 2 — ML BASELINE
# ---------------------------------------------------------------------------
with tab2:
    st.header("Crop Suitability Prediction (Random Forest — Part D Baseline)")
    st.markdown(
        "This tab reproduces your original CropWise ML pipeline. It is kept "
        "here as the **comparison baseline** for Part D — not the core system."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n2 = st.number_input("Nitrogen (N)", 0.0, 140.0, 50.0)
        p2 = st.number_input("Phosphorus (P)", 0.0, 145.0, 50.0)
    with c2:
        k2 = st.number_input("Potassium (K)", 0.0, 205.0, 50.0)
        temp2 = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
    with c3:
        hum2 = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
        ph2 = st.number_input("Soil pH", 0.0, 14.0, 6.5)
    with c4:
        rain2 = st.number_input("Rainfall (mm)", 0.0, 400.0, 100.0)
        model_choice = st.selectbox("Model", list(fitted_models.keys()))

    if st.button("Predict Best Crops"):
        model = fitted_models[model_choice]
        top_crops = predict_crop(model, scaler, label_encoder, n2, p2, k2, temp2, hum2, ph2, rain2)

        st.subheader(f"Top-3 Crop Recommendations ({model_choice})")
        for crop, prob in top_crops:
            st.write(f"**{crop}** — confidence {prob * 100:.1f}%")
            st.progress(prob)

# ---------------------------------------------------------------------------
# TAB 3 — MODEL COMPARISON
# ---------------------------------------------------------------------------
with tab3:
    st.header("Comparative Evaluation — Part B / D")
    st.markdown(
        "Random Forest, Decision Tree, and KNN evaluated on identical "
        "train/test splits. Use this table directly in your Part D results section."
    )
    st.dataframe(comparison_df, use_container_width=True)
    st.bar_chart(comparison_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score"]])

    st.markdown(
        """
        **How to use this in your report:**
        - Part B (survey): cite these three as the alternative ML paradigms you
          considered and discarded in favour of the fuzzy knowledge-based engine
          (discuss interpretability + data-dependency tradeoffs).
        - Part D (evaluation): report this table as your baseline comparison
          against the fuzzy fertilizer engine's rule-based outputs.
        """
    )