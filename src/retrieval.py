"""Query embedding and vector retrieval pipeline.

Provides two composable service functions:
  - embed_query()   — converts a text query into a vector using the controlled
                      Azure text-embedding-3-large model.
  - retrieve()      — performs similarity search against the persisted vector
                      index and returns the top-K matching passages with scores.

All model access goes through llamaindex_models.py — no direct instantiation.
The vector index is obtained via ingestion.get_index() so both modules share a
single in-memory index object.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from llama_index.core.schema import NodeWithScore

from ingestion import get_index
from llamaindex_models import get_text_embedding_3_large

DEFAULT_TOP_K = 5


# ── Service functions (no FastAPI dependency) ─────────────────────────────────


def embed_query(query: str) -> list[float]:
    """Embed a text query using the controlled Azure embedding model.

    Args:
        query: The natural language query to embed.

    Returns:
        A list of floats representing the query vector.
    """
    embed_model = get_text_embedding_3_large()
    return embed_model.get_query_embedding(query)


def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
    """Retrieve the top-K passages most similar to a query.

    Embeds the query, performs cosine similarity search against the persisted
    LlamaIndex vector store, and returns ranked results with their scores and
    source metadata — making every retrieval step observable.

    Args:
        query: Natural language query string.
        top_k: Number of top results to return.

    Returns:
        List of dicts, each containing:
          - rank          (int)   1-based rank
          - score         (float) cosine similarity score
          - text          (str)   passage text
          - passage_id    (any)   original passage identifier from metadata

    Raises:
        ValueError: If the vector index has not been ingested yet.
    """
    index = get_index()
    if index is None:
        raise ValueError(
            "Vector index is not available. "
            "Run POST /ingest to build the index first."
        )

    retriever = index.as_retriever(
        similarity_top_k=top_k,
        embed_model=get_text_embedding_3_large(),
    )
    nodes: list[NodeWithScore] = retriever.retrieve(query)

    return [
        {
            "rank": rank,
            "score": float(node.score) if node.score is not None else None,
            "text": node.node.get_content(),
            "passage_id": node.node.metadata.get("passage_id"),
        }
        for rank, node in enumerate(nodes, start=1)
    ]


# ── FastAPI router ─────────────────────────────────────────────────────────────

router = APIRouter(tags=["retrieval"])


@router.post("/retrieve")
def retrieve_endpoint(query: str, top_k: int = DEFAULT_TOP_K):
    """Retrieve the top-K passages most similar to a query.

    Performs the full retrieval pipeline:
      1. Embed the query using text-embedding-3-large
      2. Run cosine similarity search against the vector index
      3. Return ranked passages with scores

    Returns 409 if the vector index has not been ingested yet.
    """
    try:
        results = retrieve(query, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return {
        "query": query,
        "top_k": top_k,
        "results": results,
    }


@router.post("/embed")
def embed_endpoint(query: str):
    """Embed a query string and return its vector representation.

    Useful for inspecting embedding outputs and verifying that the same model
    is used for both ingestion and query-time embedding.
    Returns the vector dimensionality alongside the embedding values.
    """
    try:
        vector = embed_query(query)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Embedding model unavailable: {exc}",
        )
    return {
        "query": query,
        "dimensions": len(vector),
        "embedding": vector,
    }
