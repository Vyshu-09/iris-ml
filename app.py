from flask import Flask, request, jsonify, g
from werkzeug.exceptions import HTTPException
import joblib
import pandas as pd
import logging
import sys
import time
import uuid

# ---------- Logging configuration (stdout so Docker and Render capture it) ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("iris-ml-api")

app = Flask(__name__)

MODEL_FILE = "model.pkl"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# Only add non-sensitive features here if input-value logging is required.
SAFE_INPUT_FIELDS = set()

# ---------- Model loading with logs ----------
try:
    model = joblib.load(MODEL_FILE)
    logger.info("event=model_loaded model_file=%s", MODEL_FILE)
except Exception:
    logger.exception("event=model_load_failed model_file=%s", MODEL_FILE)
    raise


# ---------- Request logging (request id + duration + status) ----------
@app.before_request
def start_timer():
    g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    g.start = time.perf_counter()


@app.after_request
def log_request(response):
    duration_ms = (time.perf_counter() - getattr(g, "start", time.perf_counter())) * 1000
    level = logging.WARNING if response.status_code >= 400 else logging.INFO
    logger.log(
        level,
        "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        getattr(g, "request_id", "-"),
        request.method,
        request.path,
        response.status_code,
        duration_ms,
    )
    response.headers["X-Request-ID"] = getattr(g, "request_id", "-")
    return response


# ---------- Error tracking ----------
@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        return e  # 404, 405, etc. are logged as WARNING by log_request
    logger.exception("event=unhandled_exception request_id=%s path=%s",
                     getattr(g, "request_id", "-"), request.path)
    return jsonify({"error": "internal server error"}), 500


# ---------- Routes ----------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"service": "iris-species-prediction", "status": "ok", "version": "3.0"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    missing = [f for f in FEATURES if f not in data]
    if missing:
        logger.warning("event=validation_error path=%s reason=missing_fields field_locations=%s error_count=%s",
                       request.path, missing, len(missing))
        return jsonify({"error": "missing fields", "missing": missing}), 422

    try:
        values = [float(data[f]) for f in FEATURES]
    except (TypeError, ValueError):
        logger.warning("event=validation_error path=%s reason=non_numeric_value field_locations=%s error_count=1",
                       request.path, FEATURES)
        return jsonify({"error": "all values must be numbers"}), 422

    try:
        safe_input = {k: data[k] for k in FEATURES if k in SAFE_INPUT_FIELDS}
        logger.info("event=prediction_request feature_names=%s safe_input=%s",
                    sorted(FEATURES), safe_input)

        X = pd.DataFrame([values], columns=FEATURES)
        prediction = str(model.predict(X)[0])

        logger.info("event=prediction_success prediction=%s", prediction)
        return jsonify({"species": prediction})
    except Exception:
        logger.exception("event=prediction_failed request_id=%s", g.request_id)
        return jsonify({"error": "Prediction failed"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)