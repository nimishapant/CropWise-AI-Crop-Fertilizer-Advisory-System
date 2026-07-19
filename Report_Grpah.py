"""
generate_report_graphs.py
===========================
Generates all the graphs/figures you need for your written report and saves
them as PNG files in a `figures/` folder. Run this once after you've placed
the real Crop_recommendation.csv in this project folder.

Usage:
    python generate_report_graphs.py

Produces:
    figures/fuzzy_membership_N.png       - fuzzy sets for Nitrogen (Part C)
    figures/fuzzy_membership_P.png       - fuzzy sets for Phosphorus (Part C)
    figures/fuzzy_membership_K.png       - fuzzy sets for Potassium (Part C)
    figures/fuzzy_inference_example.png  - worked example: fuzzification -> output (Part C)
    figures/model_comparison_bar.png     - RF vs DT vs KNN accuracy/precision/recall/F1 (Part D)
    figures/confusion_matrix_rf.png      - Random Forest confusion matrix (Part D)
    figures/feature_importance_rf.png    - which soil/climate features matter most (Part D)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from fuzzy_fertilizer import NUTRIENT_SETS, NEED_SETS, trapezoid_membership, diagnose_soil, recommend_dosage
from baseline_models import load_dataset, train_and_evaluate, FEATURE_COLS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# 1. Fuzzy membership function plots (one per nutrient) — Part C
# ---------------------------------------------------------------------------
def plot_membership_functions(nutrient, x_max):
    fig, ax = plt.subplots(figsize=(7, 4))
    x_vals = np.linspace(0, x_max, 400)
    colors = {"Low": "#d62728", "Medium": "#ff7f0e", "High": "#2ca02c"}

    for label, params in NUTRIENT_SETS[nutrient].items():
        y_vals = [trapezoid_membership(x, params) for x in x_vals]
        ax.plot(x_vals, y_vals, label=label, color=colors[label], linewidth=2)
        ax.fill_between(x_vals, y_vals, alpha=0.15, color=colors[label])

    ax.set_xlabel(f"{nutrient} value (kg/ha)")
    ax.set_ylabel("Membership degree")
    ax.set_title(f"Fuzzy Membership Functions — {nutrient}")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3)
    save(fig, f"fuzzy_membership_{nutrient}.png")


# ---------------------------------------------------------------------------
# 2. Worked inference example — shows fuzzification -> defuzzified output
# ---------------------------------------------------------------------------
def plot_inference_example(n=25, p=60, k=110):
    diagnosis = diagnose_soil(n, p, k)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (nutrient, info) in zip(axes, diagnosis.items()):
        labels = list(info["memberships"].keys())
        degrees = list(info["memberships"].values())
        bars = ax.bar(labels, degrees, color=["#d62728", "#ff7f0e", "#2ca02c"])
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{nutrient} = {info['input_value']}\n"
                     f"Need: {info['severity']} ({info['need_score']}/100)")
        ax.set_ylabel("Membership degree")
        for bar, val in zip(bars, degrees):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.03, f"{val:.2f}", ha="center")

    fig.suptitle(f"Worked Example — Fuzzification & Inference (N={n}, P={p}, K={k})")
    fig.tight_layout()
    save(fig, "fuzzy_inference_example.png")


# ---------------------------------------------------------------------------
# 3. Model comparison bar chart — Part D
# ---------------------------------------------------------------------------
def plot_model_comparison(comparison_df):
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(comparison_df))
    width = 0.2

    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, comparison_df[metric], width, label=metric)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(comparison_df["Model"])
    ax.set_ylabel("Score")
    ax.set_title("Baseline Model Comparison (Part D)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "model_comparison_bar.png")


# ---------------------------------------------------------------------------
# 4. Confusion matrix for Random Forest — Part D
# ---------------------------------------------------------------------------
def plot_confusion_matrix(df):
    X = df[FEATURE_COLS].values
    y_raw = df["label"].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_s, y_train)
    preds = rf.predict(X_test_s)

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=90, colorbar=False)
    ax.set_title("Random Forest — Confusion Matrix")
    save(fig, "confusion_matrix_rf.png")

    return rf, le


# ---------------------------------------------------------------------------
# 5. Feature importance — Part D
# ---------------------------------------------------------------------------
def plot_feature_importance(rf_model):
    importances = rf_model.feature_importances_
    order = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [FEATURE_COLS[i] for i in order]
    ax.bar(labels, importances[order], color="#2ca02c")
    ax.set_ylabel("Importance")
    ax.set_title("Random Forest — Feature Importance")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    save(fig, "feature_importance_rf.png")


if __name__ == "__main__":
    data_path = "Crop_recommendation.csv" if os.path.exists("Crop_recommendation.csv") else None
    df = load_dataset(data_path)

    print("Generating fuzzy membership function plots...")
    plot_membership_functions("N", 140)
    plot_membership_functions("P", 145)
    plot_membership_functions("K", 205)

    print("Generating worked inference example...")
    plot_inference_example(n=25, p=60, k=110)

    print("Training baseline models and generating comparison chart...")
    comparison_df, fitted_models, scaler, le = train_and_evaluate(df)
    plot_model_comparison(comparison_df)
    comparison_df.to_csv(os.path.join(OUT_DIR, "model_comparison_table.csv"), index=False)
    print(f"Saved {OUT_DIR}/model_comparison_table.csv")

    print("Generating confusion matrix and feature importance...")
    rf_model, le2 = plot_confusion_matrix(df)
    plot_feature_importance(rf_model)

    print(f"\nAll figures saved in ./{OUT_DIR}/ — insert these directly into your report.")