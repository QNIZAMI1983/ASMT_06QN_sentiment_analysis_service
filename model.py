"""Sentiment analysis model: loading, preprocessing and prediction."""

import os
import re

MODEL_NAME = os.environ.get(
    "MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english"
)

# SST-2 is a binary classifier - it only ever emits POSITIVE or NEGATIVE. A
# neutral class is derived from confidence: anything the model is not reasonably
# sure about is reported as neutral.
NEUTRAL_THRESHOLD = float(os.environ.get("NEUTRAL_THRESHOLD", "0.65"))

# DistilBERT accepts at most 512 tokens. Truncating the raw text first stops a
# pasted essay from overwhelming the tokenizer, and bounds the cost of a request.
MAX_CHARS = int(os.environ.get("MAX_CHARS", "2000"))

_PIPELINE = None


def load_model(model_name=MODEL_NAME):
    """Load the pre-trained NLP model, reusing it across calls."""
    global _PIPELINE
    if _PIPELINE is None:
        # Imported here rather than at module scope so importing this module
        # stays cheap and tests can inject a stub pipeline.
        from transformers import pipeline

        _PIPELINE = pipeline(
            task="sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
        )
    return _PIPELINE


def preprocess(text):
    """Validate and normalise input text before inference."""
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")

    cleaned = re.sub(r"\s+", " ", text).strip()

    if not cleaned:
        raise ValueError("Input text must not be empty.")

    return cleaned[:MAX_CHARS]


def predict(text):
    """Return the sentiment of a single piece of text."""
    cleaned = preprocess(text)
    model = load_model()

    raw = model(cleaned, truncation=True)[0]
    label = raw["label"].upper()
    confidence = round(float(raw["score"]), 4)

    if confidence < NEUTRAL_THRESHOLD:
        sentiment = "neutral"
    elif label == "POSITIVE":
        sentiment = "positive"
    else:
        sentiment = "negative"

    return {
        "text": cleaned,
        "sentiment": sentiment,
        "model_label": label,
        "confidence": confidence,
    }


def predict_batch(texts):
    """Return sentiments for a list of texts."""
    if not isinstance(texts, list):
        raise ValueError("Input must be a list of strings.")
    if not texts:
        raise ValueError("Input list must not be empty.")

    return [predict(text) for text in texts]
