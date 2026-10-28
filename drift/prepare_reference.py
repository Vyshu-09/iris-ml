from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIFT_DIR = PROJECT_ROOT / "drift"

SOURCE = PROJECT_ROOT / "iris_data.csv"          # original labeled training data (no header row)
OUTPUT = DRIFT_DIR / "reference_data.csv"

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
COLUMNS = FEATURES + ["species"]

df = pd.read_csv(SOURCE, header=None, names=COLUMNS)

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

reference = df[FEATURES].dropna().copy()
reference.to_csv(OUTPUT, index=False)

print("Reference dataset created.")
print("Rows:", len(reference))
print("File:", OUTPUT)