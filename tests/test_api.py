"""API endpoint validation tests and latency benchmark."""

import time

import pytest
from sqlalchemy import text

from src.api.app import create_app
from src.config import CLASSES
from src.db.repository import get_engine


@pytest.fixture(scope="module")
def client():
    """Create a test client for the Flask app."""
    app = create_app({"TESTING": True})
    with app.test_client() as client:
        yield client


def test_index_serves_ui(client):
    """GET / -> 200 HTML page for the browser UI."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.mimetype == "text/html"
    assert b"Review Sentiment" in resp.data


def test_get_health(client):
    """GET /health -> 200, status ok."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["db"] == "ok"
    assert "model_version" in data


def test_get_model_info(client):
    """GET /model/info -> active model metadata and metrics."""
    resp = client.get("/model/info")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "model_version" in data
    assert "algorithm" in data
    assert "metrics" in data
    assert set(data["labels"]) == set(CLASSES)


def test_predict_valid(client):
    """POST /predict valid -> 200, valid label, probabilities sum to 1.0."""
    payload = {"text": "The fabric is soft and fits true to size, love it!"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    assert "X-Model-Version" in resp.headers

    data = resp.get_json()
    assert data["label"] in CLASSES
    assert 0.0 <= data["confidence"] <= 1.0
    assert abs(sum(data["probabilities"].values()) - 1.0) < 1e-3


def test_predict_empty_text(client):
    """POST /predict with empty text -> 422 Unprocessable Entity."""
    resp = client.post("/predict", json={"text": "   "})
    assert resp.status_code == 422
    data = resp.get_json()
    assert data["error"]["code"] == "validation_error"


def test_predict_missing_text(client):
    """POST /predict with missing text -> 422."""
    resp = client.post("/predict", json={"wrong_key": "some text"})
    assert resp.status_code == 422
    data = resp.get_json()
    assert data["error"]["code"] == "validation_error"


def test_predict_invalid_json(client):
    """POST /predict with invalid JSON -> 400 Bad Request."""
    resp = client.post("/predict", data="invalid{json", content_type="application/json")
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"]["code"] == "bad_request"


def test_predict_excessive_characters(client):
    """POST /predict > 5000 characters -> 422."""
    long_text = "great dress " * 600  # > 5000 chars
    resp = client.post("/predict", json={"text": long_text})
    assert resp.status_code == 422
    data = resp.get_json()
    assert data["error"]["code"] == "validation_error"


def test_predict_persists_to_database(client):
    """Verify that predictions from /predict are saved in the predictions table with source='api'."""
    test_unique_text = "Persistence test review: utterly unique phrase for verification"
    resp = client.post("/predict", json={"text": test_unique_text})
    assert resp.status_code == 200

    eng = get_engine()
    with eng.connect() as conn:
        saved = conn.execute(
            text("SELECT * FROM predictions WHERE input_text = :txt AND source = 'api'"),
            {"txt": test_unique_text},
        ).fetchone()
        assert saved is not None


def test_predict_batch_100_items(client):
    """POST /predict/batch with 100 items -> 200, count 100."""
    texts = [f"Review sample number {i}: wonderful fit and style" for i in range(100)]
    resp = client.post("/predict/batch", json={"texts": texts})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["count"] == 100
    assert len(data["predictions"]) == 100


def test_predict_batch_101_items_rejected(client):
    """POST /predict/batch with 101 items -> 422 (exceeds limit 100)."""
    texts = [f"Sample {i}" for i in range(101)]
    resp = client.post("/predict/batch", json={"texts": texts})
    assert resp.status_code == 422
    data = resp.get_json()
    assert data["error"]["code"] == "validation_error"


def test_get_reviews_pagination_cap(client):
    """GET /reviews?limit=1000 -> capped at 500."""
    resp = client.get("/reviews?limit=1000")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["limit"] == 500


def test_get_stats(client):
    """GET /stats -> aggregate counts by label and department."""
    resp = client.get("/stats")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "total_reviews" in data
    assert "by_label" in data
    assert "by_department" in data


def test_prediction_latency_benchmark(client):
    """Performance check: Sequential /predict calls locally p95 < 200 ms (NFR-02)."""
    latencies_ms = []
    sample_text = {"text": "The sweater is soft, cozy, and fits well."}

    # Run 50 sequential requests to assess p95 latency
    for _ in range(50):
        t0 = time.perf_counter()
        resp = client.post("/predict", json=sample_text)
        t1 = time.perf_counter()
        assert resp.status_code == 200
        latencies_ms.append((t1 - t0) * 1000)

    latencies_ms.sort()
    p95_idx = int(0.95 * len(latencies_ms))
    p95_latency = latencies_ms[p95_idx]
    print(
        f"\nAPI Latency Benchmark: p50={latencies_ms[len(latencies_ms) // 2]:.2f}ms, p95={p95_latency:.2f}ms"
    )
    assert p95_latency < 200.0, f"Latency check failed: p95 {p95_latency:.2f}ms >= 200ms"
