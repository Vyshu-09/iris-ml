import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
SPECIES = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"}


def test_model_loads_and_predicts():
    model = joblib.load(ROOT / "model.pkl")
    payload = json.loads((ROOT / "test_payload.json").read_text())
    X = pd.DataFrame([[payload[f] for f in FEATURES]], columns=FEATURES)
    assert str(model.predict(X)[0]) in SPECIES


def test_api_health_and_predict(monkeypatch):
    monkeypatch.chdir(ROOT)
    import app as app_module   # imported once (no reload: Prometheus metrics register at import)
    client = app_module.app.test_client()

    assert client.get("/health").status_code == 200

    payload = json.loads((ROOT / "test_payload.json").read_text())
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert any(sp in r.get_data(as_text=True) for sp in SPECIES)

    r = client.post("/predict", json={"sepal_length": 5.0})
    assert r.status_code >= 400