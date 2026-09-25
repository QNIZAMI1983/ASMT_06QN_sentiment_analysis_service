"""Tests for the sentiment analysis service.

Run with:  python -m pytest test_app.py -v
"""

import pytest

import model as sentiment_model
from app import app


class StubPipeline:
    """Stands in for transformers.pipeline('sentiment-analysis')."""

    def __call__(self, text, **kwargs):
        lowered = text.lower()
        if "meh" in lowered:
            return [{"label": "POSITIVE", "score": 0.51}]
        if any(w in lowered for w in ("love", "great", "excellent")):
            return [{"label": "POSITIVE", "score": 0.9998}]
        return [{"label": "NEGATIVE", "score": 0.9992}]


@pytest.fixture(autouse=True)
def stub_model(monkeypatch):
    """Inject the stub in place of the real model for every test."""
    monkeypatch.setattr(sentiment_model, "_PIPELINE", StubPipeline())
    yield
    monkeypatch.setattr(sentiment_model, "_PIPELINE", None)


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_home_route(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Sentiment Analysis Service" in response.data


def test_health_route(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "healthy"


def test_predict_positive(client):
    response = client.post("/predict", json={"text": "I love this product"})
    assert response.status_code == 200

    body = response.get_json()
    assert body["prediction"]["sentiment"] == "positive"
    assert body["prediction"]["confidence"] > 0.9


def test_predict_negative(client):
    response = client.post("/predict", json={"text": "This was a terrible experience"})
    assert response.get_json()["prediction"]["sentiment"] == "negative"


def test_low_confidence_becomes_neutral(client):
    response = client.post("/predict", json={"text": "meh, it was fine"})
    assert response.get_json()["prediction"]["sentiment"] == "neutral"


def test_predict_batch(client):
    response = client.post("/predict", json={"text": ["I love it", "I hate it", "meh"]})
    assert response.status_code == 200

    predictions = response.get_json()["prediction"]
    assert [p["sentiment"] for p in predictions] == ["positive", "negative", "neutral"]


def test_missing_text_key_returns_400(client):
    assert client.post("/predict", json={}).status_code == 400


def test_empty_text_returns_400(client):
    assert client.post("/predict", json={"text": "   "}).status_code == 400


def test_non_string_text_returns_400(client):
    assert client.post("/predict", json={"text": 42}).status_code == 400


def test_empty_batch_returns_400(client):
    assert client.post("/predict", json={"text": []}).status_code == 400


def test_get_on_predict_is_not_allowed(client):
    assert client.get("/predict").status_code == 405


def test_preprocess_collapses_whitespace():
    assert sentiment_model.preprocess("  hello \n\n  world  ") == "hello world"


def test_preprocess_truncates_long_input():
    long_text = "a" * 5000
    assert len(sentiment_model.preprocess(long_text)) == sentiment_model.MAX_CHARS


def test_preprocess_rejects_empty():
    with pytest.raises(ValueError):
        sentiment_model.preprocess("\n\t ")


def test_predict_returns_expected_keys():
    result = sentiment_model.predict("I love this")
    assert set(result) == {"text", "sentiment", "model_label", "confidence"}
