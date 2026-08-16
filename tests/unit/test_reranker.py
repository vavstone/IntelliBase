"""Юнит-тесты re-ranker (app/services/reranker.py).

CrossEncoder мокируется, чтобы не тянуть ~2.2 ГБ модели bge-reranker-v2-m3
в юнит-тесты. Проверяется контракт: вход — список кандидатов, выход —
пересортированный top-N с оценкой.
"""

import pytest

import app.services.reranker as reranker_module
from app.services.reranker import Reranker, top_n_doc_ids


class _FakeCrossEncoder:
    """Возвращает оценку = длина текста, чтобы порядок был предсказуемым."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def predict(self, pairs: list) -> list:
        return [float(len(text)) for (_query, text) in pairs]


@pytest.fixture
def reranker(monkeypatch) -> Reranker:
    monkeypatch.setattr(reranker_module, "CrossEncoder", _FakeCrossEncoder)
    return Reranker(model_name="fake-model", top_n=None)


def test_rerank_sorts_descending_by_score(reranker: Reranker) -> None:
    candidates = [
        {"text": "короткий", "doc_id": "a.docx"},
        {"text": "достаточно длинный текст", "doc_id": "b.docx"},
        {"text": "средний текст", "doc_id": "c.docx"},
    ]
    result = reranker.rerank("запрос", candidates)

    assert [c["doc_id"] for c in result] == ["b.docx", "c.docx", "a.docx"]
    assert all("rerank_score" in c for c in result)
    # оценка = длина текста
    assert result[0]["rerank_score"] == len("достаточно длинный текст")


def test_rerank_truncates_to_top_n(reranker: Reranker) -> None:
    candidates = [
        {"text": "один текст", "doc_id": "a.docx"},
        {"text": "два текста", "doc_id": "b.docx"},
        {"text": "три текста", "doc_id": "c.docx"},
    ]
    # явный top_n обрезает выдачу до N
    result = reranker.rerank("запрос", candidates, top_n=2)
    assert len(result) == 2


def test_rerank_preserves_original_keys(reranker: Reranker) -> None:
    candidates = [{"text": "текст", "doc_id": "x.pdf", "score": 0.9}]
    result = reranker.rerank("запрос", candidates)
    assert result[0]["doc_id"] == "x.pdf"
    assert result[0]["score"] == 0.9
    assert "rerank_score" in result[0]


def test_rerank_empty_candidates(reranker: Reranker) -> None:
    assert reranker.rerank("запрос", []) == []


def test_top_n_doc_ids_returns_ordered_ids(reranker: Reranker) -> None:
    candidates = [
        {"text": "короткий", "doc_id": "a.docx"},
        {"text": "очень длинный текст для победы", "doc_id": "b.docx"},
    ]
    ids = top_n_doc_ids("запрос", candidates, reranker, top_n=2)
    assert ids == ["b.docx", "a.docx"]
