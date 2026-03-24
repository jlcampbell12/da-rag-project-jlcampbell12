"""Phase 2 retrieval tests.

Test tiers:
  - Unit / wiring  : import checks, router registration (no network)
  - Endpoint       : FastAPI TestClient, /retrieve and /embed structure checks
                     (no index available — verifies 409 handling)
  - Azure          : real embedding calls and real retrieval against live index
                     (requires Azure auth + ingested index)

Run with: uv run pytest
"""

import pytest
from fastapi.testclient import TestClient


# ── Import / wiring tests (no network) ───────────────────────────────────────


def test_retrieval_module_imports():
    from retrieval import DEFAULT_TOP_K, embed_query, retrieve
    assert callable(embed_query)
    assert callable(retrieve)
    assert isinstance(DEFAULT_TOP_K, int)
    assert DEFAULT_TOP_K > 0


def test_retrieval_router_registered_on_app():
    from main import app
    routes = {r.path for r in app.routes}
    assert "/retrieve" in routes
    assert "/embed" in routes


# ── Endpoint structure tests (TestClient, no Azure needed) ───────────────────


@pytest.fixture(scope="module")
def client():
    from main import app
    with TestClient(app) as c:
        yield c


def test_retrieve_without_index_returns_409(client):
    """If no index has been ingested, /retrieve must return 409."""
    import ingestion

    # Temporarily clear any cached in-memory index so the endpoint has no index
    original_index = ingestion._index
    ingestion._index = None

    # Also point INDEX_DIR somewhere that doesn't exist
    import retrieval
    from pathlib import Path
    original_dir = ingestion.INDEX_DIR
    ingestion.INDEX_DIR = Path("/nonexistent/path/for/testing")

    response = client.post("/retrieve?query=test+query")

    # Restore state
    ingestion._index = original_index
    ingestion.INDEX_DIR = original_dir

    assert response.status_code == 409


def test_retrieve_endpoint_exists(client):
    """POST /retrieve must exist (not 404/405) regardless of index state."""
    response = client.post("/retrieve?query=hello")
    assert response.status_code != 404
    assert response.status_code != 405


def test_embed_endpoint_exists(client):
    """POST /embed must exist regardless of Azure auth state."""
    response = client.post("/embed?query=hello")
    assert response.status_code != 404
    assert response.status_code != 405


# ── Azure integration tests (require auth + ingested index) ──────────────────


def test_embed_query_returns_list_of_floats():
    """embed_query must call Azure and return a non-empty list of floats.

    Requires: Azure auth + network
    """
    from retrieval import embed_query

    vector = embed_query("What is Uruguay?")
    assert isinstance(vector, list)
    assert len(vector) > 0
    assert all(isinstance(v, float) for v in vector)


def test_embed_query_is_consistent():
    """Two calls with the same text should return the same vector."""
    from retrieval import embed_query

    v1 = embed_query("capital of France")
    v2 = embed_query("capital of France")
    assert v1 == v2


def test_embed_query_vectors_differ_for_different_queries():
    """Different queries should produce meaningfully different vectors."""
    from retrieval import embed_query

    v1 = embed_query("What is the capital of France?")
    v2 = embed_query("How many legs does a spider have?")
    assert v1 != v2


def test_embed_endpoint_returns_correct_structure(client):
    """POST /embed with Azure auth should return query, dimensions, embedding."""
    response = client.post("/embed?query=test+embedding")
    assert response.status_code == 200
    body = response.json()
    assert "query" in body
    assert "dimensions" in body
    assert "embedding" in body
    assert body["dimensions"] > 0
    assert len(body["embedding"]) == body["dimensions"]


def test_embed_has_expected_dimensionality(client):
    """text-embedding-3-large produces 3072-dimensional vectors."""
    response = client.post("/embed?query=test")
    assert response.status_code == 200
    assert response.json()["dimensions"] == 3072


def test_retrieve_returns_results_when_index_ready(client):
    """POST /retrieve must return a ranked list of passages.

    Requires: Azure auth + ingested index (run POST /ingest?limit=50 first)
    """
    response = client.post("/retrieve?query=What+is+Uruguay%3F&top_k=3")
    assert response.status_code == 200
    body = response.json()
    assert "query" in body
    assert "top_k" in body
    assert "results" in body
    assert len(body["results"]) == 3


def test_retrieve_results_have_required_fields(client):
    response = client.post("/retrieve?query=capital+city&top_k=2")
    assert response.status_code == 200
    for result in response.json()["results"]:
        assert "rank" in result
        assert "score" in result
        assert "text" in result
        assert "passage_id" in result


def test_retrieve_results_are_ranked_descending(client):
    """Results should be returned highest-score first."""
    response = client.post("/retrieve?query=South+America&top_k=5")
    assert response.status_code == 200
    scores = [r["score"] for r in response.json()["results"] if r["score"] is not None]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_respects_top_k(client):
    for k in (1, 3, 5):
        response = client.post(f"/retrieve?query=river&top_k={k}")
        assert response.status_code == 200
        assert len(response.json()["results"]) == k


def test_retrieve_results_have_non_empty_text(client):
    response = client.post("/retrieve?query=agriculture&top_k=3")
    assert response.status_code == 200
    for result in response.json()["results"]:
        assert len(result["text"]) > 0
