# CropWise — AI Crop & Fertilizer Advisory System

An AI decision-support tool for farmers with two parts:

1. **Fuzzy logic fertilizer engine** (core system) — recommends fertilizer
   type and dosage from soil N/P/K readings using fuzzy rules, not ML.
2. **ML baseline** — Random Forest, Decision Tree, and KNN crop predictors,
   used as a comparison, not the main system.

## Files

- `fuzzy_fertilizer.py` — core AI engine (fuzzy rules + inference)
- `baseline_models.py` — ML models for comparison
- `app.py` — Streamlit web app
- `requirements.txt` — dependencies


Add the Kaggle **Crop Recommendation Dataset** as `Crop_recommendation.csv`
in this folder (columns: `N, P, K, temperature, humidity, ph, rainfall,
label`). Without it, the app uses synthetic demo data.

## Why fuzzy logic is the core system, not ML

The fuzzy engine reasons using explicit rules and handles imprecise
readings directly — no training data needed. Random Forest/DT/KNN are
kept as the baseline comparison for the report.