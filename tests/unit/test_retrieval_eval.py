"""Юнит-тесты retrieval-метрик (app/services/retrieval_eval.py).

Проверяют чистую логику на синтетических списках doc_id — без Qdrant/эмбеддингов.
"""

from app.services.retrieval_eval import (
    evaluate_retrieval,
    hit_rate_at_k,
    mrr_at_k,
    recall_at_k,
)


# ── hit_rate_at_k ────────────────────────────────────────────────────────

def test_hit_rate_found_in_top_k() -> None:
    assert hit_rate_at_k(["a", "b", "c"], ["c"], k=5) == 1.0


def test_hit_rate_missed() -> None:
    assert hit_rate_at_k(["a", "b", "c"], ["z"], k=5) == 0.0


def test_hit_rate_beyond_k_is_miss() -> None:
    # релевантный документ на 6-й позиции не считается при k=5
    assert hit_rate_at_k(["a", "b", "c", "d", "e", "f"], ["f"], k=5) == 0.0


def test_hit_rate_no_relevant() -> None:
    assert hit_rate_at_k(["a"], [], k=5) == 0.0


# ── mrr_at_k ─────────────────────────────────────────────────────────────

def test_mrr_rank_one() -> None:
    assert mrr_at_k(["a", "b", "c"], ["a"], k=10) == 1.0


def test_mrr_rank_three() -> None:
    assert mrr_at_k(["a", "b", "c"], ["c"], k=10) == 1.0 / 3


def test_mrr_missed() -> None:
    assert mrr_at_k(["a", "b"], ["z"], k=10) == 0.0


def test_mrr_first_occurrence_of_multiple_relevant() -> None:
    assert mrr_at_k(["x", "a", "a"], ["a"], k=10) == 0.5


# ── recall_at_k ──────────────────────────────────────────────────────────

def test_recall_all_found() -> None:
    assert recall_at_k(["a", "b", "c"], ["a", "b"], k=10) == 1.0


def test_recall_partial() -> None:
    assert recall_at_k(["a", "c"], ["a", "b"], k=10) == 0.5


def test_recall_no_relevant() -> None:
    assert recall_at_k(["a"], [], k=10) == 0.0


# ── evaluate_retrieval ───────────────────────────────────────────────────

def _fake_retrieve(mapping: dict) -> callable:
    def retrieve(question: str) -> list[str]:
        return mapping.get(question, [])

    return retrieve


def test_evaluate_aggregates_three_metrics() -> None:
    golden = [
        {"question": "q1", "relevant_doc_ids": ["a"]},
        {"question": "q2", "relevant_doc_ids": ["b"]},
    ]
    retrieve = _fake_retrieve({"q1": ["a", "x"], "q2": ["y", "z"]})

    result = evaluate_retrieval(retrieve, golden, top_k=10)

    # q1: hit=1, mrr=1/1, recall=1; q2: hit=0, mrr=0, recall=0
    assert result["hit_rate_at_5"] == 0.5
    assert result["mrr_at_10"] == 0.5
    assert result["recall_at_10"] == 0.5
    assert result["n_questions"] == 2
    assert len(result["per_question"]) == 2


def test_evaluate_empty_golden() -> None:
    result = evaluate_retrieval(lambda q: [], [], top_k=10)
    assert result["n_questions"] == 0
    assert result["hit_rate_at_5"] == 0.0
