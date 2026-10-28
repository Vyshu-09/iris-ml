from pathlib import Path
import argparse
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIFT_DIR = PROJECT_ROOT / "drift"

SOURCE = PROJECT_ROOT / "iris_data.csv"
REFERENCE = DRIFT_DIR / "reference_data.csv"
OUTPUT = DRIFT_DIR / "current_data.csv"                    # features only -> drift detection
OUTPUT_LABELED = DRIFT_DIR / "current_labeled_data.csv"    # features + species -> retraining (classroom only)

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["stable", "drift"], default="stable")
args = parser.parse_args()

rng = np.random.default_rng(42)
reference = pd.read_csv(REFERENCE)
full = pd.read_csv(SOURCE, header=None, names=FEATURES + [TARGET])

n = min(100, len(full))
current = full.sample(n=n, replace=True, random_state=42).reset_index(drop=True)

# Iris measurements are in cm (about 0.1 - 7.9), so noise and shift are scaled
# to each feature's standard deviation instead of the fixed +-1 / -15 values
# used for the 0-100 student marks.
std = reference[FEATURES].std()

# Small random noise for both modes (5 % of the feature std)
for f in FEATURES:
    current[f] = current[f] + rng.normal(0, 0.05 * std[f], size=len(current))

if args.mode == "drift":
    # Classroom-only synthetic shift: the new batch is measured with a different
    # instrument/population, so every measurement is about 25 % larger.
    # (A uniform scale keeps the species structure, so retraining stays meaningful.)
    for f in FEATURES:
        current[f] = current[f] * 1.25

for f in FEATURES:
    current[f] = current[f].clip(lower=0.1).round(2)

current[FEATURES].to_csv(OUTPUT, index=False)
current[FEATURES + [TARGET]].to_csv(OUTPUT_LABELED, index=False)

print("Mode:", args.mode)
print("Rows:", len(current))
print("File:", OUTPUT)
print("Labeled classroom batch:", OUTPUT_LABELED)