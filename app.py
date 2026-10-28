from flask import Flask, request, jsonify, g, Response
from werkzeug.exceptions import HTTPException
import joblib
import pandas as pd
import logging
import sys
import time
import uuid
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

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

# ---------- Prometheus metrics (Practical 12) ----------
HTTP_REQUESTS = Counter(
    "iris_api_http_requests_total",
    "Total HTTP requests received by the Iris API",
    ["method", "path", "status"],
)
HTTP_LATENCY = Histogram(
    "iris_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)
PREDICTIONS = Counter(
    "iris_api_predictions_total",
    "Total Iris prediction attempts",
    ["result"],
)
MODEL_LOADED = Gauge(
    "iris_api_model_loaded",
    "Whether the Iris model is loaded",
)

# ---------- Model loading with logs ----------
try:
    model = joblib.load(MODEL_FILE)
    MODEL_LOADED.set(1)
    logger.info("event=model_loaded model_file=%s", MODEL_FILE)
except Exception:
    MODEL_LOADED.set(0)
    logger.exception("event=model_load_failed model_file=%s", MODEL_FILE)
    raise


# ---------- Request logging + metrics (request id + duration + status) ----------
@app.before_request
def start_timer():
    g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    g.start = time.perf_counter()


@app.after_request
def log_request(response):
    duration = time.perf_counter() - getattr(g, "start", time.perf_counter())
    duration_ms = duration * 1000

    # Metrics: skip Prometheus own scrapes; group unknown URLs under "unmatched"
    if request.path != "/metrics":
        metric_path = request.url_rule.rule if request.url_rule else "unmatched"
        HTTP_REQUESTS.labels(
            method=request.method,
            path=metric_path,
            status=str(response.status_code),
        ).inc()
        HTTP_LATENCY.labels(method=request.method, path=metric_path).observe(duration)

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


@app.route("/metrics", methods=["GET"])
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


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

        PREDICTIONS.labels(result="success").inc()
        logger.info("event=prediction_success prediction=%s", prediction)
        return jsonify({"species": prediction})
    except Exception:
        PREDICTIONS.labels(result="error").inc()
        logger.exception("event=prediction_failed request_id=%s", g.request_id)
        return jsonify({"error": "Prediction failed"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
