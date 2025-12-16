import json
import logging
from typing import Dict, List, Tuple, Union

import joblib
import numpy as np
from flask import Flask, jsonify, request

MODEL_PATH = "linear_mmm_model.pkl"
FEATURES_PATH = "mmm_model_features.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


def load_features(path: str) -> List[str]:
    with open(path, "r") as f:
        return json.load(f)


def load_model(path: str):
    return joblib.load(path)


try:
    feature_names = load_features(FEATURES_PATH)
    model = load_model(MODEL_PATH)
    model_load_error = None
    logger.info("Model and features loaded.")
except Exception as exc:  # pragma: no cover - defensive guard
    feature_names = []
    model = None
    model_load_error = str(exc)
    logger.exception("Failed to load model or features.")


def validate_payload(data: Union[Dict, None]) -> Tuple[List[float], List[str]]:
    errors: List[str] = []

    if not isinstance(data, dict):
        errors.append("Invalid JSON body; expected an object.")
        return [], errors

    missing = [f for f in feature_names if f not in data]
    if missing:
        errors.append(f"Missing required fields: {', '.join(missing)}")

    extras = [k for k in data.keys() if k not in feature_names]
    if extras:
        errors.append(f"Unexpected fields: {', '.join(extras)}")

    values: List[float] = []
    for feat in feature_names:
        raw_val = data.get(feat)
        if raw_val is None:
            errors.append(f"Field '{feat}' is required and must be numeric.")
            continue
        try:
            values.append(float(raw_val))
        except (TypeError, ValueError):
            errors.append(f"Field '{feat}' must be numeric.")

    return values, errors


@app.route("/health", methods=["GET"])
def health():
    status = {
        "status": "ok" if model and not model_load_error else "error",
        "model_loaded": model is not None,
        "feature_count": len(feature_names),
        "error": model_load_error,
    }
    code = 200 if status["status"] == "ok" else 503
    return jsonify(status), code


@app.route("/")
def home():
    return "MMM Model is running!"


@app.route("/predict", methods=["POST"])
def predict():
    if model_load_error or model is None:
        return (
            jsonify({"error": "Model not loaded", "detail": model_load_error}),
            503,
        )

    data = request.get_json(silent=True)
    values, errors = validate_payload(data)
    if errors:
        return jsonify({"errors": errors}), 400

    X = np.array(values, dtype=float).reshape(1, -1)
    try:
        prediction = float(model.predict(X)[0])
    except Exception as exc:  # pragma: no cover - runtime guard
        logger.exception("Prediction failed.")
        return jsonify({"error": "Prediction failed", "detail": str(exc)}), 500

    return jsonify({"predicted_revenue": prediction})


if __name__ == "__main__":
    logger.info("Starting Flask server...")
    app.run(debug=True, use_reloader=False)