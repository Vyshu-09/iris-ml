import pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier

cols = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
df = pd.read_csv("iris_data.csv", header=None, names=cols)

model = RandomForestClassifier(random_state=42)
model.fit(df[cols[:4]], df["species"])
joblib.dump(model, "model.pkl")
print("Model saved. Rows used:", len(df))
