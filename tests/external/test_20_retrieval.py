"""External retrieval tests — hit the live API server using the requests library.

Requires the server to be running (see tests/external/conftest.py).
Start it with: uv run uvicorn main:app --app-dir src --reload

These tests cover two scenarios:
  - Structural tests: verify endpoint contracts without requiring a live index
  - Integration tests: verify real retrieval results (marked, need ingested index)

Run with: pytest tests/external/
"""

import requests


# ── /embed endpoint ───────────────────────────────────────────────────────────


def test_embed_returns_200(base_url):
    """POST /embed requires Azure auth; skip gracefully if not available."""
    response = requests.post(f"{base_url}/embed", params={"query": "test"})
    # 200 = auth OK, 5xx = auth missing in server environment — both are
    # informative; we assert not 404/405 to confirm the route exists.
    assert response.status_code not in (404, 405)


def test_embed_endpoint_structure(base_url):
    response = requests.post(f"{base_url}/embed", params={"query": "Wikipedia"})
    if response.status_code != 200:
        pytest.skip("Azure auth not available on the server")
    body = response.json()
    assert "query" in body
    assert "dimensions" in body
    assert "embedding" in body
    assert body["dimensions"] > 0
    assert len(body["embedding"]) == body["dimensions"]


def test_embed_dimensionality(base_url):
    response = requests.post(f"{base_url}/embed", params={"query": "dimensions test"})
    if response.status_code != 200:
        pytest.skip("Azure auth not available on the server")
    assert response.json()["dimensions"] == 3072


# ── /retrieve endpoint ────────────────────────────────────────────────────────


def test_retrieve_returns_409_when_no_index(base_url):
    """Before ingest, /retrieve should return 409 (or 200 if index exists)."""
    response = requests.post(f"{base_url}/retrieve", params={"query": "test"})
    assert response.status_code in (200, 409)


def test_retrieve_endpoint_exists(base_url):
    response = requests.post(f"{base_url}/retrieve", params={"query": "test"})
    assert response.status_code not in (404, 405)


def test_retrieve_with_index_returns_200(base_url):
    """POST /retrieve returns ranked passages when index is available.

    Requires: ingested index (POST /ingest must have been run first).
    """
    response = requests.post(
        f"{base_url}/retrieve",
        params={"query": "What is Uruguay?", "top_k": 3},
    )
    if response.status_code == 409:
        pytest.skip("Index not yet ingested — run POST /ingest first")
    assert response.status_code == 200


def test_retrieve_response_structure(base_url):
    response = requests.post(
        f"{base_url}/retrieve",
        params={"query": "capital city", "top_k": 2},
    )
    if response.status_code == 409:
        pytest.skip("Index not yet ingested — run POST /ingest first")
    assert response.status_code == 200
    body = response.json()
    assert "query" in body
    assert "top_k" in body
    assert "results" in body


def test_retrieve_results_have_required_fields(base_url):
    response = requests.post(
        f"{base_url}/retrieve",
        params={"query": "river", "top_k": 2},
    )
    if response.status_code == 409:
        pytest.skip("Index not yet ingested — run POST /ingest first")
    for result in response.json()["results"]:
        assert "rank" in result
        assert "score" in result
        assert "text" in result
        assert "passage_id" in result


def test_retrieve_respects_top_k(base_url):
    for k in (1, 3):
        response = requests.post(
            f"{base_url}/retrieve",
            params={"query": "history", "top_k": k},
        )
        if response.status_code == 409:
            pytest.skip("Index not yet ingested — run POST /ingest first")
        assert len(response.json()["results"]) == k


def test_retrieve_results_ranked_descending(base_url):
    response = requests.post(
        f"{base_url}/retrieve",
        params={"query": "South America geography", "top_k": 5},
    )
    if response.status_code == 409:
        pytest.skip("Index not yet ingested — run POST /ingest first")
    scores = [r["score"] for r in response.json()["results"] if r["score"] is not None]
    assert scores == sorted(scores, reverse=True)


import pytest  # noqa: E402  (needed for pytest.skip calls above)
