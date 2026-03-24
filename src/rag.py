"""Augmented generation pipeline (Phase 3).

Given a user query this module:
  1. Retrieves the top-K relevant passages via retrieval.retrieve()
  2. Constructs an explicit, inspectable augmented prompt
  3. Sends the prompt to GPT-4o via the controlled interface
  4. Returns the answer together with every intermediate artefact so that
     notebooks and tests can observe the full trace.

All model access goes through llamaindex_models.py — no direct instantiation.

Evaluation helpers:
  - load_test_questions()  loads the HuggingFace test Q&A parquet
  - evaluate()             runs a batch of questions and scores exact-match
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException

from llamaindex_models import get_gpt4o
from retrieval import retrieve

TEST_QUESTIONS_URL = (
    "hf://datasets/rag-datasets/rag-mini-wikipedia"
    "/data/test.parquet/part.0.parquet"
)

# ── Prompt template ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly based on the "
    "provided context passages. If the answer is not contained in the context, "
    "say \"I don't have enough information to answer that.\""
)

_PROMPT_TEMPLATE = """\
Use the following context passages to answer the question.

Context:
{context}

Question: {question}

Answer:"""


# ── Service functions (no FastAPI dependency) ─────────────────────────────────


def build_prompt(question: str, passages: list[dict[str, Any]]) -> str:
    """Construct the augmented prompt from a question and retrieved passages.

    Keeping prompt construction as a pure function makes it fully testable
    and observable without any LLM calls.

    Args:
        question: The original user question.
        passages: List of dicts with at least a 'text' key (as returned by
                  retrieval.retrieve()).

    Returns:
        The complete prompt string that will be sent to the LLM.
    """
    context_block = "\n\n".join(
        f"[{p['rank']}] {p['text']}" for p in passages
    )
    return _PROMPT_TEMPLATE.format(context=context_block, question=question)


def answer_question(
    question: str,
    top_k: int = 5,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> dict[str, Any]:
    """Run the full RAG pipeline for a single question.

    Returns a dict containing every intermediate artefact so each step is
    fully observable:
      - question        original query
      - retrieved       list of top-K passages with rank/score/text/passage_id
      - prompt          the full augmented prompt sent to the LLM
      - answer          the generated answer string
      - model           the model name used

    Args:
        question:    Natural language question.
        top_k:       Number of passages to retrieve.
        temperature: LLM temperature (0.0 = deterministic).
        max_tokens:  Max tokens for the LLM response.

    Raises:
        ValueError: If the vector index is not available.
    """
    # Step 1: Retrieve relevant passages
    retrieved = retrieve(question, top_k=top_k)

    # Step 2: Build augmented prompt (pure function — no LLM)
    prompt = build_prompt(question, retrieved)

    # Step 3: Call GPT-4o through the controlled interface
    llm = get_gpt4o(
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=_SYSTEM_PROMPT,
    )
    response = llm.complete(prompt)
    answer = response.text.strip()

    return {
        "question": question,
        "retrieved": retrieved,
        "prompt": prompt,
        "answer": answer,
        "model": "gpt-4o",
    }


def load_test_questions(url: str = TEST_QUESTIONS_URL) -> pd.DataFrame:
    """Load the HuggingFace test Q&A parquet.

    Returns a DataFrame with at minimum 'question' and 'answer' columns.
    """
    return pd.read_parquet(url)


def evaluate(
    questions: list[str],
    expected_answers: list[str],
    top_k: int = 5,
) -> dict[str, Any]:
    """Run a batch of questions through the RAG pipeline and score them.

    Uses case-insensitive substring match as a simple recall metric:
    a result is counted correct if the expected answer appears anywhere
    in the generated answer.

    Args:
        questions:        List of question strings.
        expected_answers: Corresponding expected answer strings.
        top_k:            Passages to retrieve per question.

    Returns:
        Dict with:
          - total           number of questions evaluated
          - correct         number of substring-match correct answers
          - accuracy        correct / total
          - results         list of per-question dicts with full trace
    """
    results = []
    correct = 0

    for question, expected in zip(questions, expected_answers):
        try:
            result = answer_question(question, top_k=top_k)
            is_correct = expected.lower() in result["answer"].lower()
            if is_correct:
                correct += 1
            results.append({**result, "expected": expected, "correct": is_correct})
        except Exception as exc:
            results.append({
                "question": question,
                "expected": expected,
                "answer": None,
                "correct": False,
                "error": str(exc),
            })

    total = len(questions)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0.0,
        "results": results,
    }


# ── FastAPI router ─────────────────────────────────────────────────────────────

router = APIRouter(tags=["rag"])


@router.post("/query")
def query_endpoint(question: str, top_k: int = 5):
    """Full RAG pipeline: retrieve relevant passages then generate an answer.

    Returns the answer together with the retrieved passages and the augmented
    prompt so every step is observable.

    Returns 409 if the vector index has not been ingested yet.
    Returns 503 if the LLM is unavailable (auth or network issue).
    """
    try:
        result = answer_question(question, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {exc}")
    return result


@router.post("/query/debug")
def query_debug_endpoint(question: str, top_k: int = 5):
    """Same as /query but also returns the full prompt sent to the LLM.

    Useful for notebook-driven observability — you can inspect exactly what
    context was injected and how the prompt was constructed.
    """
    try:
        result = answer_question(question, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {exc}")
    return result


@router.get("/test-questions")
def test_questions_endpoint(offset: int = 0, limit: int = 10):
    """Browse the HuggingFace test Q&A pairs used for evaluation.

    Useful for inspecting what evaluation questions look like before running
    a full evaluation pass.
    """
    df = load_test_questions()
    total = len(df)
    page = df.iloc[offset: offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "questions": [
            {
                "id": int(idx),
                "question": str(row["question"]),
                "answer": str(row["answer"]),
            }
            for idx, row in page.iterrows()
        ],
    }
