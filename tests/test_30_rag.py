"""Phase 3 RAG tests.

Test tiers:
  - Pure / wiring  : build_prompt(), router registration, test Q&A data loading
                     (no Azure, no index)
  - Endpoint       : TestClient — verifies 409 when no index, route existence
  - Azure          : full end-to-end answer generation + evaluation
                     (requires Azure auth + ingested index)

Run with: uv run pytest
"""

import pytest
from fastapi.testclient import TestClient


# ── Import / wiring tests (no network) ───────────────────────────────────────


def test_rag_module_imports():
    from rag import answer_question, build_prompt, evaluate, load_test_questions
    assert callable(build_prompt)
    assert callable(answer_question)
    assert callable(load_test_questions)
    assert callable(evaluate)


def test_rag_router_registered_on_app():
    from main import app
    routes = {r.path for r in app.routes}
    assert "/query" in routes
    assert "/query/debug" in routes
    assert "/test-questions" in routes


# ── build_prompt tests (pure function — no network) ───────────────────────────


def test_build_prompt_contains_question():
    from rag import build_prompt
    passages = [{"rank": 1, "text": "Some passage about Uruguay."}]
    prompt = build_prompt("What is Uruguay?", passages)
    assert "What is Uruguay?" in prompt


def test_build_prompt_contains_passage_text():
    from rag import build_prompt
    passages = [{"rank": 1, "text": "Uruguay is in South America."}]
    prompt = build_prompt("test question", passages)
    assert "Uruguay is in South America." in prompt


def test_build_prompt_includes_all_passages():
    from rag import build_prompt
    passages = [
        {"rank": 1, "text": "First passage."},
        {"rank": 2, "text": "Second passage."},
        {"rank": 3, "text": "Third passage."},
    ]
    prompt = build_prompt("question", passages)
    assert "First passage." in prompt
    assert "Second passage." in prompt
    assert "Third passage." in prompt


def test_build_prompt_includes_rank_labels():
    from rag import build_prompt
    passages = [{"rank": 1, "text": "text"}, {"rank": 2, "text": "text2"}]
    prompt = build_prompt("q", passages)
    assert "[1]" in prompt
    assert "[2]" in prompt


def test_build_prompt_with_empty_passages():
    from rag import build_prompt
    prompt = build_prompt("What is Python?", [])
    assert "What is Python?" in prompt
    assert isinstance(prompt, str)


# ── Test Q&A data loading (HuggingFace — needs network) ──────────────────────


def test_load_test_questions_returns_dataframe():
    import pandas as pd
    from rag import load_test_questions
    df = load_test_questions()
    assert isinstance(df, pd.DataFrame)


def test_load_test_questions_not_empty():
    from rag import load_test_questions
    df = load_test_questions()
    assert len(df) > 0


def test_load_test_questions_has_question_column():
    from rag import load_test_questions
    df = load_test_questions()
    assert "question" in df.columns


def test_load_test_questions_has_answer_column():
    from rag import load_test_questions
    df = load_test_questions()
    assert "answer" in df.columns


# ── Endpoint tests via TestClient (no Azure needed) ───────────────────────────


@pytest.fixture(scope="module")
def client():
    from main import app
    with TestClient(app) as c:
        yield c


def test_query_without_index_returns_409(client):
    """Before ingestion, /query must return 409."""
    import ingestion
    from pathlib import Path

    original_index = ingestion._index
    original_dir = ingestion.INDEX_DIR
    ingestion._index = None
    ingestion.INDEX_DIR = Path("/nonexistent/path")

    response = client.post("/query?question=test")

    ingestion._index = original_index
    ingestion.INDEX_DIR = original_dir

    assert response.status_code == 409


def test_query_endpoint_exists(client):
    response = client.post("/query?question=test")
    assert response.status_code not in (404, 405)


def test_query_debug_endpoint_exists(client):
    response = client.post("/query/debug?question=test")
    assert response.status_code not in (404, 405)


def test_test_questions_endpoint_returns_200(client):
    response = client.get("/test-questions?limit=3")
    assert response.status_code == 200


def test_test_questions_endpoint_structure(client):
    response = client.get("/test-questions?limit=3")
    body = response.json()
    assert "total" in body
    assert "questions" in body
    assert len(body["questions"]) == 3


def test_test_questions_items_have_required_fields(client):
    response = client.get("/test-questions?limit=2")
    for q in response.json()["questions"]:
        assert "id" in q
        assert "question" in q
        assert "answer" in q


# ── Azure end-to-end integration tests (auth + index required) ────────────────


def test_answer_question_returns_answer():
    """Full RAG pipeline: retrieve + generate. Requires Azure auth + index."""
    from rag import answer_question
    result = answer_question("What is Uruguay?", top_k=3)
    assert "answer" in result
    assert len(result["answer"]) > 0


def test_answer_question_returns_retrieved_passages():
    from rag import answer_question
    result = answer_question("What is Uruguay?", top_k=3)
    assert "retrieved" in result
    assert len(result["retrieved"]) == 3


def test_answer_question_returns_prompt():
    from rag import answer_question
    result = answer_question("What is Uruguay?", top_k=3)
    assert "prompt" in result
    assert "What is Uruguay?" in result["prompt"]


def test_answer_question_returns_model_name():
    from rag import answer_question
    result = answer_question("What is Uruguay?", top_k=3)
    assert result["model"] == "gpt-4o"


def test_query_endpoint_with_live_index(client):
    """POST /query returns full trace when index is available."""
    response = client.post("/query?question=What+is+Uruguay%3F&top_k=3")
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "retrieved" in body
    assert "prompt" in body
    assert len(body["answer"]) > 0


def test_query_debug_endpoint_with_live_index(client):
    response = client.post("/query/debug?question=Where+is+Montevideo%3F&top_k=3")
    assert response.status_code == 200
    body = response.json()
    assert "prompt" in body
    assert "answer" in body


# ── Evaluation tests (requires Azure auth + index + test.parquet) ─────────────


def test_evaluate_small_batch():
    """Run 3 questions from the test set through the full pipeline and score."""
    from rag import evaluate, load_test_questions

    df = load_test_questions()
    sample = df.head(3)
    questions = sample["question"].tolist()
    answers = sample["answer"].tolist()

    result = evaluate(questions, answers, top_k=5)

    assert result["total"] == 3
    assert 0.0 <= result["accuracy"] <= 1.0
    assert len(result["results"]) == 3
    for r in result["results"]:
        assert "question" in r
        assert "answer" in r
        assert "correct" in r
