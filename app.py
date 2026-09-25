"""Flask microservice exposing a pre-trained sentiment analysis model."""

import os

from flask import Flask, jsonify, request

import model as sentiment_model

app = Flask(__name__)


@app.route("/")
def home():
    return "<h3>Sentiment Analysis Service</h3><p>POST JSON to /predict</p>"


@app.route("/health")
def health():
    """Health check for the load balancer. Deliberately does not touch the model
    so a cold container can report healthy before the weights finish loading."""
    return jsonify({"status": "healthy"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """Return the sentiment of the submitted text.

    Accepts either:
        {"text": "I love this product"}
        {"text": ["great service", "terrible support"]}
    """
    payload = request.get_json(silent=True)

    if not payload or "text" not in payload:
        return (
            jsonify(
                {
                    "error": "Request body must be JSON containing a 'text' key.",
                    "status": "error",
                }
            ),
            400,
        )

    text = payload["text"]

    try:
        if isinstance(text, list):
            prediction = sentiment_model.predict_batch(text)
        else:
            prediction = sentiment_model.predict(text)
    except ValueError as exc:
        return jsonify({"error": str(exc), "status": "error"}), 400
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {exc}", "status": "error"}), 503

    return jsonify({"prediction": prediction, "status": "success"})


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=debug)
