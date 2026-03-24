"""External ingestion tests — hit the live API server using the requests library.

Requires the server to be running (see tests/external/conftest.py).
Start it with: uv run uvicorn main:app --app-dir src --reload

Run with: pytest tests/external/
"""

import requests


# ── /ingest/status endpoint ───────────────────────────────────────────────────


def test_ingest_status_returns_200(base_url):
    response = requests.get(f"{base_url}/ingest/status")
    assert response.status_code == 200


def test_ingest_status_content_type_is_json(base_url):
    response = requests.get(f"{base_url}/ingest/status")
    assert "application/json" in response.headers.get("content-type", "")


def test_ingest_status_has_required_fields(base_url):
    body = requests.get(f"{base_url}/ingest/status").json()
    assert "status" in body
    assert "passage_count" in body
    assert "index_on_disk" in body
    assert "index_path" in body


def test_ingest_status_value_is_valid(base_url):
    body = requests.get(f"{base_url}/ingest/status").json()
    assert body["status"] in ("not_ingested", "ingesting", "ready")


# ── /passages endpoint ────────────────────────────────────────────────────────


def test_passages_returns_200(base_url):
    response = requests.get(f"{base_url}/passages")
    assert response.status_code == 200


def test_passages_has_required_fields(base_url):
    body = requests.get(f"{base_url}/passages?limit=2").json()
    assert "total" in body
    assert "offset" in body
    assert "limit" in body
    assert "passages" in body


def test_passages_total_is_positive(base_url):
    body = requests.get(f"{base_url}/passages?limit=1").json()
    assert body["total"] > 0


def test_passages_respects_limit_param(base_url):
    body = requests.get(f"{base_url}/passages?limit=4").json()
    assert len(body["passages"]) == 4


def test_passages_respects_offset_param(base_url):
    p0 = requests.get(f"{base_url}/passages?limit=1&offset=0").json()["passages"][0]
    p1 = requests.get(f"{base_url}/passages?limit=1&offset=1").json()["passages"][0]
    assert p0["id"] != p1["id"]


def test_passages_items_have_id_and_text(base_url):
    passages = requests.get(f"{base_url}/passages?limit=3").json()["passages"]
    for p in passages:
        assert "id" in p
        assert "text" in p
        assert len(p["text"]) > 0


def test_passages_default_limit_is_ten(base_url):
    body = requests.get(f"{base_url}/passages").json()
    assert len(body["passages"]) == 10
