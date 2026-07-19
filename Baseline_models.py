"""
baseline_models.py
====================
PART D BASELINE / COMPARISON — NOT the core AI system.

The brief requires you to "compare your implemented approach against at
least one baseline or alternative method" (Part D) and to survey >=3
approaches in Part B. This module trains the ML classifiers your original
CropWise draft already used (Random Forest, Decision Tree, KNN) and
re-purposes them as the comparison baseline, not the headline system.

Usage:
    python baseline_models.py path/to/Crop_recommendation.csv

Expected CSV columns (standard Kaggle "Crop Recommendation Dataset"):
    N, P, K, temperature, humidity, ph, rainfall, label
"""

import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COL = "label"


def generate_synthetic_dataset(n_samples=1200, seed=42):
    """
    Fallback synthetic dataset generator — used only if the user hasn't
    supplied the real Crop_recommendation.csv yet, so the pipeline can still
    be demoed end-to-end. Replace with the real Kaggle CSV for your actual
    report numbers.
    """
    rng = np.random.default_rng(seed)
    crops = ["rice", "maize", "chickpea", "banana", "mango", "cotton"]
    rows = []
    for crop in crops:
        n = 200 if len(rows) == 0 else n_samples // len(crops)
        for _ in range(n):
            rows.append({
                "N": rng.uniform(0, 140),
                "P": rng.uniform(5, 145),
                "K": rng.uniform(5, 205),
                "temperature": rng.uniform(10, 40),
                "humidity": rng.uniform(20, 100),
                "ph": rng.uniform(4, 9),
                "rainfall": rng.uniform(20, 300),
                "label": crop,
            })
    return pd.DataFrame(rows)


def load_dataset(path=None):
    if path:
        df = pd.read_csv(path)
        missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing expected columns: {missing}")
        return df
    print("[WARNING] No dataset path given — using synthetic demo data. "
          "Download the real 'Crop Recommendation Dataset' from Kaggle for your report.")
    return generate_synthetic_dataset()


def train_and_evaluate(df, test_size=0.2, cv_folds=5, random_state=42):
    """Trains RF, Decision Tree, and KNN on the same split and returns a
    comparison table (this feeds directly into your Part D results table)."""
    X = df[FEATURE_COLS].values
    y_raw = df[TARGET_COL].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    }

    results = []
    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)

        cv_scores = cross_val_score(model, X_train_s, y_train, cv=cv_folds)

        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, preds), 4),
            "Precision": round(precision_score(y_test, preds, average="weighted", zero_division=0), 4),
            "Recall": round(recall_score(y_test, preds, average="weighted", zero_division=0), 4),
            "F1-Score": round(f1_score(y_test, preds, average="weighted", zero_division=0), 4),
            f"CV Mean ({cv_folds}-fold)": round(cv_scores.mean(), 4),
        })
        fitted_models[name] = model

    comparison_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
    return comparison_df, fitted_models, scaler, le


def predict_crop(model, scaler, label_encoder, n, p, k, temperature, humidity, ph, rainfall, top_k=3):
    """Predict top-k most likely crops for a given set of inputs using a
    fitted model (used by the Streamlit app's baseline tab)."""
    X = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
    X_s = scaler.transform(X)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_s)[0]
        top_idx = np.argsort(probs)[::-1][:top_k]
        return [(label_encoder.inverse_transform([i])[0], round(float(probs[i]), 3)) for i in top_idx]
    else:
        pred = model.predict(X_s)[0]
        return [(label_encoder.inverse_transform([pred])[0], 1.0)]


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    data = load_dataset(csv_path)
    comparison, models, scaler, le = train_and_evaluate(data)
    print("\n=== Part D: Baseline Model Comparison ===")
    print(comparison.to_string(index=False))