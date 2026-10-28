from pathlib import Path
import json
import pandas as pd
from scipy.stats import ks_2samp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIFT_DIR = PROJECT_ROOT / "drift"

REFERENCE = DRIFT_DIR / "reference_data.csv"
CURRENT = DRIFT_DIR / "current_data.csv"
REPORT = DRIFT_DIR / "drift_report.json"

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

P_VALUE_THRESHOLD = 0.05
MIN_DRIFTED_FEATURES = 2

reference = pd.read_csv(REFERENCE)
current = pd.read_csv(CURRENT)

results = {}
drifted_features = []

for feature in FEATURES:
    ref_values = reference[feature].dropna()
    cur_values = current[feature].dropna()

    statistic, p_value = ks_2samp(ref_values, cur_values)
    drift_detected = bool(p_value < P_VALUE_THRESHOLD)

    results[feature] = {
        "ks_statistic": round(float(statistic), 6),
        "p_value": round(float(p_value), 6),
        "drift_detected": drift_detected,
        "reference_mean": round(float(ref_values.mean()), 4),
        "current_mean": round(float(cur_values.mean()), 4),
    }
    if drift_detected:
        drifted_features.append(feature)

overall_drift = len(drifted_features) >= MIN_DRIFTED_FEATURES

report = {
    "method": "two-sample KS test",
    "p_value_threshold": P_VALUE_THRESHOLD,
    "min_drifted_features": MIN_DRIFTED_FEATURES,
    "drifted_feature_count": len(drifted_features),
    "drifted_features": drifted_features,
    "overall_drift": overall_drift,
    "features": results,
}

with open(REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
print("\nDrift report:", REPORT)