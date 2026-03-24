"""Phase 1 ingestion tests.

Tests are ordered so that non-Azure tests run first, followed by the full
Azure integration test that actually builds the vector index.

Test tiers:
  - Data loading    : downloads passages from HuggingFace (network required)
  - Transformation  : pure Python / pandas (no network)
  - Endpoint        : FastAPI TestClient, /passages loads from HF (network)
  - Ingestion       : calls Azure text-embedding-3-large (Azure auth required)

Run with: uv run pytest
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ── Data loading tests (HuggingFace — needs network) ─────────────────────────


def test_passages_parquet_loads_from_hf():
    """Passages parquet should download from HuggingFace without error."""
    from ingestion import load_passages
    df = load_passages()
    assert isinstance(df, pd.DataFrame)


def test_passages_dataframe_not_empty():
    from ingestion import load_passages
    df = load_passages()
    assert len(df) > 0


def test_passages_have_passage_column():
    from ingestion import TEXT_COLUMN, load_passages
    df = load_passages()
    assert TEXT_COLUMN in df.columns, (
        f"Expected column '{TEXT_COLUMN}'. Actual columns: {list(df.columns)}"
    )


def test_passages_id_is_dataframe_index():
    """The real parquet uses 'id' as the DataFrame index, not a regular column."""
    from ingestion import load_passages
    df = load_passages()
    assert df.index.name == "id" or "id" not in df.columns  # id is the index
    assert len(df.index) > 0


def test_passages_text_is_non_empty_strings():
    from ingestion import TEXT_COLUMN, load_passages
    df = load_passages()
    sample = df[TEXT_COLUMN].head(10)
    assert sample.notna().all()
    assert (sample.str.len() > 0).all()


# ── Document conversion tests (no network needed) ────────────────────────────


def test_passages_to_documents_returns_documents():
    from llama_index.core import Document

    from ingestion import passages_to_documents

    sample = pd.DataFrame({"id": [1, 2], "passage": ["text one", "text two"]})
    docs = passages_to_documents(sample)
    assert all(isinstance(d, Document) for d in docs)


def test_passages_to_documents_correct_count():
    from ingestion import passages_to_documents

    sample = pd.DataFrame({"id": [1, 2, 3], "passage": ["a", "b", "c"]})
    docs = passages_to_documents(sample)
    assert len(docs) == 3


def test_passages_to_documents_preserves_text():
    from ingestion import passages_to_documents

    sample = pd.DataFrame({"id": [42], "passage": ["Hello Wikipedia"]})
    docs = passages_to_documents(sample)
    assert docs[0].text == "Hello Wikipedia"


def test_passages_to_documents_includes_passage_id_in_metadata():
    """passage_id in metadata should come from the DataFrame index."""
    from ingestion import passages_to_documents

    # Match the real schema: id is the index, passage is the only column
    sample = pd.DataFrame({"passage": ["some text"]}, index=[99])
    sample.index.name = "id"
    docs = passages_to_documents(sample)
    assert docs[0].metadata["passage_id"] == 99


# ── Endpoint tests via TestClient ─────────────────────────────────────────────


@pytest.fixture(scope="module")
def client():
    from main import app
    with TestClient(app) as c:
        yield c


def test_ingest_status_initial_state(client):
    """Before any ingest call, status should be 'not_ingested' or 'ready'
    (ready if a previous test session already built the index on disk)."""
    response = client.get("/ingest/status")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("not_ingested", "ready")
    assert "passage_count" in body
    assert "index_on_disk" in body


def test_ingest_status_has_index_path(client):
    response = client.get("/ingest/status")
    assert "index_path" in response.json()


def test_passages_endpoint_returns_200(client):
    response = client.get("/passages?limit=5")
    assert response.status_code == 200


def test_passages_endpoint_returns_passages_list(client):
    response = client.get("/passages?limit=5")
    body = response.json()
    assert "passages" in body
    assert isinstance(body["passages"], list)


def test_passages_endpoint_respects_limit(client):
    response = client.get("/passages?limit=3")
    body = response.json()
    assert len(body["passages"]) == 3


def test_passages_endpoint_respects_offset(client):
    r1 = client.get("/passages?limit=1&offset=0").json()
    r2 = client.get("/passages?limit=1&offset=1").json()
    assert r1["passages"][0]["id"] != r2["passages"][0]["id"]


def test_passages_endpoint_returns_total_count(client):
    response = client.get("/passages?limit=1")
    body = response.json()
    assert body["total"] > 0


def test_passages_have_id_and_text_fields(client):
    response = client.get("/passages?limit=2")
    for p in response.json()["passages"]:
        assert "id" in p
        assert "text" in p
        assert len(p["text"]) > 0


# ── Full ingestion integration test (requires Azure auth) ─────────────────────


def test_run_ingestion_with_small_limit(tmp_path):
    """Build and persist a real vector index from 5 passages using Azure embeddings.

    Requires: Azure auth  (`azd auth login --scope api://ailab/Model.Access`)
    Uses a tmp_path to avoid overwriting the production index.
    """
    from ingestion import run_ingestion

    result = run_ingestion(limit=5, index_dir=tmp_path / "test_index")

    assert result["status"] == "ready"
    assert result["passage_count"] == 5
    assert result["index_on_disk"] is True


def test_index_persisted_to_disk_after_ingestion(tmp_path):
    """After ingestion, the index directory should contain LlamaIndex store files."""
    from ingestion import run_ingestion

    index_dir = tmp_path / "test_index_disk"
    run_ingestion(limit=3, index_dir=index_dir)

    assert index_dir.exists()
    # LlamaIndex persists at least a docstore and vector store JSON
    persisted_files = list(index_dir.iterdir())
    assert len(persisted_files) > 0


def test_ingest_endpoint_with_limit(client, tmp_path, monkeypatch):
    """POST /ingest?limit=3 should return status=ready.

    Monkeypatches INDEX_DIR to use a temp path so we don't overwrite production.
    Requires Azure auth.
    """
    import ingestion

    monkeypatch.setattr(ingestion, "INDEX_DIR", tmp_path / "endpoint_test_index")
    response = client.post("/ingest?limit=3")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["passage_count"] == 3
