"""Smoke tests: verify imports, app creation, and /health endpoint.

These tests use FastAPI's TestClient — no running server required.
Run with: uv run pytest
"""

import pytest
from fastapi.testclient import TestClient


# ── Import / registry tests ───────────────────────────────────────────────────


def test_main_app_imports():
    """FastAPI app should be importable without errors."""
    from main import app
    assert app is not None
    assert app.title == "Wikipedia RAG System"


def test_llamaindex_models_imports():
    """llamaindex_models module and all public symbols should import cleanly."""
    from llamaindex_models import (
        MODEL_REGISTRY,
        ModelAccessError,
        get_available_models,
        get_chat_model,
        get_embedding_model,
        get_gpt4o,
        get_text_embedding_3_large,
    )
    assert callable(get_chat_model)
    assert callable(get_embedding_model)
    assert callable(get_gpt4o)
    assert callable(get_text_embedding_3_large)
    assert callable(get_available_models)
    assert MODEL_REGISTRY is not None
    assert ModelAccessError is not None


def test_model_registry_has_gpt4o():
    from llamaindex_models import MODEL_REGISTRY
    assert "gpt-4o" in MODEL_REGISTRY["chat"]


def test_model_registry_has_embedding_model():
    from llamaindex_models import MODEL_REGISTRY
    assert "text-embedding-3-large" in MODEL_REGISTRY["embeddings"]


def test_unknown_chat_model_raises_model_access_error():
    """Requesting an unregistered chat model must raise ModelAccessError."""
    from llamaindex_models import ModelAccessError, get_chat_model
    with pytest.raises(ModelAccessError):
        get_chat_model("nonexistent-model-xyz")


def test_unknown_embedding_model_raises_model_access_error():
    """Requesting an unregistered embedding model must raise ModelAccessError."""
    from llamaindex_models import ModelAccessError, get_embedding_model
    with pytest.raises(ModelAccessError):
        get_embedding_model("nonexistent-embedding-xyz")


# ── /health endpoint tests via TestClient ────────────────────────────────────


@pytest.fixture(scope="module")
def client():
    from main import app
    with TestClient(app) as c:
        yield c


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_content_type_is_json(client):
    response = client.get("/health")
    assert "application/json" in response.headers["content-type"]


def test_health_status_is_ok(client):
    response = client.get("/health")
    assert response.json()["status"] == "ok"


def test_health_has_service_field(client):
    response = client.get("/health")
    assert "service" in response.json()


def test_health_has_version_field(client):
    response = client.get("/health")
    assert "version" in response.json()
