"""External RAG tests — hit the live API server using the requests library.

Requires the server to be running (see tests/external/conftest.py).
Start it with: uv run uvicorn main:app --app-dir src --reload

Tests skip gracefully when the index is not ingested or Azure auth is absent.

Run with: pytest tests/external/
"""

import pytest
import requests


# ── /test-questions endpoint ──────────────────────────────────────────────────


def test_test_questions_returns_200(base_url):
    response = requests.get(f"{base_url}/test-questions?limit=3")
    assert response.status_code == 200


def test_test_questions_has_required_fields(base_url):
    body = requests.get(f"{base_url}/test-questions?limit=3").json()
    assert "total" in body
    assert "questions" in body
    assert "offset" in body
    assert "limit" in body


def test_test_questions_total_is_positive(base_url):
    body = requests.get(f"{base_url}/test-questions?limit=1").json()
    assert body["total"] > 0


def test_test_questions_respects_limit(base_url):
    body = requests.get(f"{base_url}/test-questions?limit=4").json()
    assert len(body["questions"]) == 4


def test_test_questions_items_have_required_fields(base_url):
    questions = requests.get(f"{base_url}/test-questions?limit=2").json()["questions"]
    for q in questions:
        assert "id" in q
        assert "question" in q
        assert "answer" in q


# ── /query endpoint ───────────────────────────────────────────────────────────


def test_query_endpoint_exists(base_url):
    response = requests.post(f"{base_url}/query", params={"question": "test"})
    assert response.status_code not in (404, 405)


def test_query_returns_200_or_409(base_url):
    response = requests.post(f"{base_url}/query", params={"question": "What is Uruguay?"})
    assert response.status_code in (200, 409, 503)


def test_query_returns_full_trace(base_url):
    response = requests.post(
        f"{base_url}/query",
        params={"question": "What is Uruguay?", "top_k": 3},
    )
    if response.status_code in (409, 503):
        pytest.skip("Index not ingested or Azure auth unavailable")
    assert response.status_code == 200
    body = response.json()
    assert "question" in body
    assert "answer" in body
    assert "retrieved" in body
    assert "prompt" in body


def test_query_answer_is_non_empty(base_url):
    response = requests.post(
        f"{base_url}/query",
        params={"question": "What is Uruguay?", "top_k": 3},
    )
    if response.status_code in (409, 503):
        pytest.skip("Index not ingested or Azure auth unavailable")
    body = response.json()
    assert len(body["answer"]) > 0


def test_query_retrieved_count_matches_top_k(base_url):
    k = 3
    response = requests.post(
        f"{base_url}/query",
        params={"question": "capital city", "top_k": k},
    )
    if response.status_code in (409, 503):
        pytest.skip("Index not ingested or Azure auth unavailable")
    assert len(response.json()["retrieved"]) == k


def test_query_prompt_contains_question(base_url):
    question = "Where is Montevideo?"
    response = requests.post(
        f"{base_url}/query",
        params={"question": question, "top_k": 3},
    )
    if response.status_code in (409, 503):
        pytest.skip("Index not ingested or Azure auth unavailable")
    assert question in response.json()["prompt"]


# ── /query/debug endpoint ─────────────────────────────────────────────────────


def test_query_debug_endpoint_exists(base_url):
    response = requests.post(f"{base_url}/query/debug", params={"question": "test"})
    assert response.status_code not in (404, 405)


def test_query_debug_returns_prompt(base_url):
    response = requests.post(
        f"{base_url}/query/debug",
        params={"question": "What is Uruguay?", "top_k": 3},
    )
    if response.status_code in (409, 503):
        pytest.skip("Index not ingested or Azure auth unavailable")
    assert "prompt" in response.json()
