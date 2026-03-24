"""Knowledge ingestion pipeline.

Loads Wikipedia passages from HuggingFace, converts them to LlamaIndex Documents,
builds a VectorStoreIndex with the controlled Azure embedding model, and persists
the index to disk for later retrieval and querying.

All model access goes through llamaindex_models.py — no direct instantiation.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)

from llamaindex_models import get_text_embedding_3_large

# Prevent LlamaIndex from attempting to instantiate a default OpenAI LLM;
# all model access is explicit and controlled via llamaindex_models.py.
Settings.llm = None  # type: ignore[assignment]

PASSAGES_URL = (
    "hf://datasets/rag-datasets/rag-mini-wikipedia"
    "/data/passages.parquet/part.0.parquet"
)
INDEX_DIR = Path(__file__).parent.parent / "data" / "vector_store"
TEXT_COLUMN = "passage"

# ── Module-level state ────────────────────────────────────────────────────────

_lock = threading.Lock()
_passages: pd.DataFrame | None = None
_index: VectorStoreIndex | None = None
_state: dict = {
    "status": "not_ingested",   # "not_ingested" | "ingesting" | "ready"
    "passage_count": 0,
    "index_path": str(INDEX_DIR),
}


# ── Service functions (no FastAPI dependency) ─────────────────────────────────


def load_passages(url: str = PASSAGES_URL) -> pd.DataFrame:
    """Download and module-level-cache the passages parquet from HuggingFace."""
    global _passages
    if _passages is None:
        _passages = pd.read_parquet(url)
    return _passages


def passages_to_documents(df: pd.DataFrame) -> list[Document]:
    """Convert a passages DataFrame to a list of LlamaIndex Document objects.

    The rag-mini-wikipedia passages parquet uses the row index as the passage
    identifier (the index name is 'id' but it is not a regular column).
    """
    return [
        Document(
            text=str(row[TEXT_COLUMN]),
            metadata={"passage_id": idx},
        )
        for idx, row in df.iterrows()
    ]


def get_status() -> dict:
    """Return the current ingestion status including whether the index is on disk."""
    with _lock:
        status = dict(_state)
    status["index_on_disk"] = (Path(status["index_path"])).exists()
    return status


def run_ingestion(
    limit: int | None = None,
    index_dir: Path | None = None,
) -> dict:
    """Run the full ingestion pipeline.

    Downloads passages from HuggingFace, embeds them with the controlled Azure
    text-embedding-3-large model, builds a VectorStoreIndex, and persists it.

    Args:
        limit: If set, ingest only the first N passages. Useful for testing.
        index_dir: Override the default index persistence directory. Useful for
            testing — avoids writing into the production index directory.

    Returns:
        Status dict after ingestion completes.

    Raises:
        RuntimeError: If ingestion is already in progress.
    """
    global _index

    target_dir = index_dir or INDEX_DIR

    with _lock:
        if _state["status"] == "ingesting":
            raise RuntimeError("Ingestion already in progress.")
        _state["status"] = "ingesting"
        _state["passage_count"] = 0

    try:
        df = load_passages()
        if limit is not None:
            df = df.head(limit)

        documents = passages_to_documents(df)
        embed_model = get_text_embedding_3_large()

        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model,
            show_progress=True,
        )

        target_dir.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(target_dir))

        with _lock:
            _index = index
            _state["status"] = "ready"
            _state["passage_count"] = len(df)
            _state["index_path"] = str(target_dir)

    except Exception:
        with _lock:
            _state["status"] = "not_ingested"
        raise

    return get_status()


def get_index(index_dir: Path | None = None) -> VectorStoreIndex | None:
    """Return the in-memory index, loading from disk if the index dir exists.

    Returns None if neither an in-memory nor persisted index is available.
    """
    global _index
    if _index is not None:
        return _index
    target_dir = index_dir or INDEX_DIR
    if not target_dir.exists():
        return None
    embed_model = get_text_embedding_3_large()
    storage_context = StorageContext.from_defaults(persist_dir=str(target_dir))
    _index = load_index_from_storage(storage_context, embed_model=embed_model)
    with _lock:
        _state["status"] = "ready"
    return _index


# ── FastAPI router ────────────────────────────────────────────────────────────

router = APIRouter(tags=["ingestion"])


@router.post("/ingest")
def ingest_endpoint(limit: int | None = None):
    """Trigger the ingestion pipeline.

    Downloads passages from HuggingFace, generates embeddings via the controlled
    Azure text-embedding-3-large model, and persists the vector index to disk.

    Use `?limit=N` to ingest only N passages — recommended during development
    and testing to keep run times short and Azure API usage low.

    Returns 409 if ingestion is already running.
    """
    try:
        return run_ingestion(limit=limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get("/ingest/status")
def ingest_status_endpoint():
    """Return current ingestion state."""
    return get_status()


@router.get("/passages")
def passages_endpoint(offset: int = 0, limit: int = 10):
    """Browse raw passages from the HuggingFace dataset.

    Supports pagination via `offset` and `limit` query parameters.
    Useful for inspecting the source data before or after ingestion.
    """
    df = load_passages()
    total = len(df)
    page = df.iloc[offset: offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "passages": [
            {
                "id": idx,
                "text": str(row[TEXT_COLUMN]),
            }
            for idx, row in page.iterrows()
        ],
    }
