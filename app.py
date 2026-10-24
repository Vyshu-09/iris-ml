from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("model.pkl")
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


@app.route("/", methods=["GET"])
def home():
    return jsonify({"service": "iris-species-prediction", "status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": "missing fields", "missing": missing}), 400
    try:
        X = pd.DataFrame([[float(data[f]) for f in FEATURES]], columns=FEATURES)
    except (TypeError, ValueError):
        return jsonify({"error": "all values must be numbers"}), 400
    return jsonify({"species": str(model.predict(X)[0])})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
