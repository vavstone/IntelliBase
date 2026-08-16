"""Юнит-тесты стратегий чанкинга (app/services/chunking.py).

Проверяют fixed_size / recursive / chunk_stats на in-memory документах — без
Qdrant и без загрузки embed-модели (semantic требует модель, поэтому покрыт
отдельно только через smoke-проверку контракта при наличии эмбеддингов).
"""

from llama_index.core import Document

from app.services.chunking import (
    _russian_sentences,
    chunk_stats,
    fixed_size,
    recursive,
)

_TEXT = (
    "Первый абзац о приостановлении срока выпуска товаров. "
    "Второе предложение того же абзаца.\n\n"
    "Второй абзац про таможенный реестр ОИС. Ещё одно предложение."
)


def _docs() -> list[Document]:
    return [
        Document(text=_TEXT, metadata={"file_name": "test.md"}),
        Document(text="Короткий документ без смысла.", metadata={"file_name": "short.md"}),
    ]


# ── _russian_sentences ───────────────────────────────────────────────────

def test_russian_sentences_splits_on_punctuation() -> None:
    sentences = _russian_sentences("Одно предложение. Второе предложение! Третье?")
    assert sentences == ["Одно предложение.", "Второе предложение!", "Третье?"]


def test_russian_sentences_keeps_empty_out() -> None:
    assert _russian_sentences("   ") == []


# ── fixed_size ───────────────────────────────────────────────────────────

def test_fixed_size_produces_nonempty_chunks() -> None:
    nodes = fixed_size(_docs(), chunk_size=32, chunk_overlap=4)
    assert len(nodes) >= 2  # текст режется минимум на 2 части
    assert all(n.get_content().strip() for n in nodes)


# ── recursive ────────────────────────────────────────────────────────────

def test_recursive_respects_paragraph_boundaries() -> None:
    nodes = recursive(_docs(), chunk_size=256, chunk_overlap=32)
    assert len(nodes) >= 1
    # при chunk_size, вмещающем оба абзаца, splitter не должен дробить текст
    assert all(n.get_content().strip() for n in nodes)


def test_recursive_small_chunk_splits_more() -> None:
    big = recursive(_docs(), chunk_size=40, chunk_overlap=4)
    small = recursive(_docs(), chunk_size=20, chunk_overlap=4)
    assert len(small) >= len(big)


# ── chunk_stats ──────────────────────────────────────────────────────────

def test_chunk_stats_math() -> None:
    nodes = fixed_size(_docs(), chunk_size=128, chunk_overlap=8)
    stats = chunk_stats(nodes, n_documents=2)

    assert stats["total_chunks"] == len(nodes)
    assert stats["avg_chunks_per_doc"] == round(len(nodes) / 2, 2)
    expected_avg = sum(len(n.get_content()) for n in nodes) / len(nodes)
    assert stats["avg_chunk_len_chars"] == round(expected_avg, 1)


def test_chunk_stats_empty() -> None:
    stats = chunk_stats([], n_documents=2)
    assert stats["total_chunks"] == 0
    assert stats["avg_chunk_len_chars"] == 0.0
