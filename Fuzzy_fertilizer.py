"""
fuzzy_fertilizer.py
====================
CORE AI SYSTEM (Part C) — Knowledge-Based Fertilizer Recommendation Engine
using Fuzzy Logic (Mamdani-style inference).

Why this is the "core" and not Random Forest:
- Knowledge representation -> explicit fuzzy sets + IF-THEN rules (not learned weights)
- Reasoning under uncertainty -> fuzzy membership functions handle vague/imprecise
  soil readings instead of hard thresholds
- System architecture -> Fuzzifier -> Inference Engine -> Defuzzifier pipeline

This module has NO dependency on scikit-fuzzy so it will run anywhere pandas/numpy runs.
All membership functions are implemented from scratch (triangular / trapezoidal),
which also makes it easy to explain and diagram in your report.
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. KNOWLEDGE BASE: fuzzy set definitions for each nutrient (Nitrogen,
#    Phosphorus, Potassium). Ranges are based on typical agronomic values
#    (kg/ha) seen in soil test reports and the Kaggle Crop Recommendation
#    dataset's N/P/K columns.
# ---------------------------------------------------------------------------

# Each fuzzy set is defined as a trapezoid (a, b, c, d):
#   membership rises from 0 at a to 1 at b, stays 1 until c, falls to 0 at d.
NUTRIENT_SETS = {
    "N": {  # Nitrogen kg/ha
        "Low":    (0, 0, 30, 60),
        "Medium": (40, 60, 80, 100),
        "High":   (80, 110, 140, 140),
    },
    "P": {  # Phosphorus kg/ha
        "Low":    (0, 0, 20, 45),
        "Medium": (30, 50, 70, 90),
        "High":   (70, 100, 145, 145),
    },
    "K": {  # Potassium kg/ha
        "Low":    (0, 0, 30, 60),
        "Medium": (40, 70, 100, 130),
        "High":   (100, 150, 205, 205),
    },
}

# Output fuzzy sets for "fertilizer need" (0-100 need-score scale), shared
# across Urea (-> N), DAP (-> P), MOP (-> K).
NEED_SETS = {
    "Low":    (0, 0, 20, 40),
    "Medium": (25, 45, 55, 75),
    "High":   (60, 80, 100, 100),
}

FERTILIZER_FOR_NUTRIENT = {"N": "Urea", "P": "DAP", "K": "MOP (Muriate of Potash)"}

# Rule base (this IS the explicit knowledge representation):
#   IF nutrient is Low    THEN fertilizer_need is High
#   IF nutrient is Medium THEN fertilizer_need is Medium
#   IF nutrient is High   THEN fertilizer_need is Low
RULES = {
    "Low": "High",
    "Medium": "Medium",
    "High": "Low",
}


def trapezoid_membership(x, params):
    """Compute membership degree of x in a trapezoidal fuzzy set (a,b,c,d)."""
    a, b, c, d = params
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a) if b != a else 1.0
    if c < x < d:
        return (d - x) / (d - c) if d != c else 1.0
    return 0.0


def fuzzify(value, fuzzy_sets):
    """Step 1: Fuzzification — map a crisp input value to membership degrees
    across all fuzzy sets (Low/Medium/High) for that variable."""
    return {label: trapezoid_membership(value, params) for label, params in fuzzy_sets.items()}


def infer_need(memberships):
    """Step 2: Inference — apply the rule base. For each input fuzzy label with
    nonzero membership, activate the corresponding output rule (min-implication),
    then aggregate output memberships per output label using max (standard
    Mamdani aggregation)."""
    aggregated = {"Low": 0.0, "Medium": 0.0, "High": 0.0}
    for input_label, degree in memberships.items():
        if degree <= 0:
            continue
        output_label = RULES[input_label]
        aggregated[output_label] = max(aggregated[output_label], degree)
    return aggregated


def defuzzify_centroid(aggregated, output_sets=NEED_SETS, resolution=200):
    """Step 3: Defuzzification via centroid (center of gravity) method.
    Builds the aggregated output fuzzy surface across a discretized universe
    (0-100 need score) and returns its centroid as the crisp output."""
    universe = np.linspace(0, 100, resolution)
    aggregated_curve = np.zeros_like(universe)

    for label, activation in aggregated.items():
        if activation <= 0:
            continue
        a, b, c, d = output_sets[label]
        set_curve = np.array([trapezoid_membership(x, (a, b, c, d)) for x in universe])
        clipped = np.minimum(set_curve, activation)  # clip membership at rule activation level
        aggregated_curve = np.maximum(aggregated_curve, clipped)

    if aggregated_curve.sum() == 0:
        return 0.0
    centroid = (universe * aggregated_curve).sum() / aggregated_curve.sum()
    return float(centroid)


def need_to_label(score):
    """Convert a 0-100 crisp need-score back into a human-readable severity band."""
    if score < 35:
        return "Low"
    elif score < 65:
        return "Medium"
    else:
        return "High"


def diagnose_soil(n, p, k):
    """
    Full pipeline for one nutrient reading set: fuzzify -> infer -> defuzzify.
    Returns a dict per nutrient with: crisp need score, severity label,
    recommended fertilizer, and the raw membership degrees (useful for
    displaying the fuzzification step in your UI/report).
    """
    results = {}
    for nutrient, value in [("N", n), ("P", p), ("K", k)]:
        memberships = fuzzify(value, NUTRIENT_SETS[nutrient])
        aggregated = infer_need(memberships)
        need_score = defuzzify_centroid(aggregated)
        severity = need_to_label(need_score)
        results[nutrient] = {
            "input_value": value,
            "memberships": memberships,
            "need_score": round(need_score, 1),
            "severity": severity,
            "fertilizer": FERTILIZER_FOR_NUTRIENT[nutrient],
        }
    return results


def recommend_dosage(need_score, nutrient):
    """
    Simple linear mapping from crisp need-score (0-100) to a suggested
    dosage in kg/ha, bounded by realistic agronomic application ranges.
    This turns the fuzzy diagnosis into an actionable recommendation,
    which is the whole point of a decision-support system.
    """
    dosage_ranges = {
        "N": (0, 120),   # Urea equivalent kg/ha
        "P": (0, 90),    # DAP equivalent kg/ha
        "K": (0, 100),   # MOP equivalent kg/ha
    }
    low, high = dosage_ranges[nutrient]
    return round(low + (need_score / 100) * (high - low), 1)


if __name__ == "__main__":
    # Quick manual test
    example = diagnose_soil(n=25, p=60, k=110)
    for nutrient, info in example.items():
        dosage = recommend_dosage(info["need_score"], nutrient)
        print(f"{nutrient}: severity={info['severity']}, need_score={info['need_score']}, "
              f"recommend {dosage} kg/ha of {info['fertilizer']}")