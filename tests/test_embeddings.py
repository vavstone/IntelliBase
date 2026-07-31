"""Smoke-тесты для EmbeddingsService.

Проверяют:
1. Генерацию эмбеддингов (sentence-transformers, без API-ключа)
2. Попадание в кеш при повторном вызове
3. Нормализацию векторов (||v|| ≈ 1.0)
4. Косинусную близость на парах из mini-benchmark
5. Методы embed_query / embed_documents для E5-моделей

Для запуска: pytest tests/test_embeddings.py -v
"""

import json
import math
import pytest
from pathlib import Path

from app.services.embeddings import (
    EmbeddingsService,
    create_embeddings_service,
)


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture
def service_st() -> EmbeddingsService:
    """Сервис на sentence-transformers — работает без сети и API-ключа."""
    return create_embeddings_service(
        provider="sentence_transformers",
        model="all-MiniLM-L6-v2",
    )


@pytest.fixture
def mini_benchmark() -> list[dict]:
    """Загружает мини-бенчмарк из tests/eval/mini_benchmark.json (первичный)
    или eval/mini_benchmark.json (запасной)."""
    path = Path(__file__).parent / "eval" / "mini_benchmark.json"
    if not path.exists():
        path = Path(__file__).parent.parent / "eval" / "mini_benchmark.json"
    if not path.exists():
        pytest.skip("mini_benchmark.json not found")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) >= 5, f"Expected 5+ pairs, got {len(data)}"
    return data


# ---------------------------------------------------------------------------
# Тесты
# ---------------------------------------------------------------------------

class TestEmbeddingsBasic:
    """Базовые проверки генерации эмбеддингов."""

    @pytest.mark.asyncio
    async def test_embed_one_returns_vector(self, service_st: EmbeddingsService):
        """embed_one возвращает список float."""
        vec = await service_st.embed_one("тестовый текст")
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert all(isinstance(x, float) for x in vec)

    @pytest.mark.asyncio
    async def test_embed_texts_returns_same_count(self, service_st: EmbeddingsService):
        """embed_texts возвращает столько же векторов, сколько текстов."""
        texts = ["первый", "второй", "третий"]
        vecs = await service_st.embed_texts(texts)
        assert len(vecs) == len(texts)
        assert all(len(v) == len(vecs[0]) for v in vecs)

    @pytest.mark.asyncio
    async def test_embed_texts_empty(self, service_st: EmbeddingsService):
        """Пустой список — пустой результат."""
        assert await service_st.embed_texts([]) == []


class TestNormalization:
    """Проверка нормализации векторов."""

    @pytest.mark.asyncio
    async def test_vectors_are_normalized(self, service_st: EmbeddingsService):
        """Все возвращаемые векторы имеют норму ≈ 1.0."""
        texts = ["короткий", "длинный текст с большим количеством разных слов"]
        vecs = await service_st.embed_texts(texts)
        for vec in vecs:
            norm = math.sqrt(sum(x * x for x in vec))
            assert abs(norm - 1.0) < 0.01, f"Norm = {norm}, expected ~1.0"


class TestCache:
    """Проверка кеширования."""

    @pytest.mark.asyncio
    async def test_repeat_call_hits_cache(self, service_st: EmbeddingsService):
        """Повторный вызов embed_texts с теми же текстами попадает в кеш."""
        texts = ["текст для кеша A", "текст для кеша B"]

        # Первый вызов — все промахи
        await service_st.embed_texts(texts)
        stats1 = service_st.cache_stats()
        assert stats1["misses"] == 2

        # Повторный — все попадания
        await service_st.embed_texts(texts)
        stats2 = service_st.cache_stats()
        assert stats2["hits"] == 2
        assert stats2["misses"] == 2  # не изменилось

    @pytest.mark.asyncio
    async def test_partial_cache_hit(self, service_st: EmbeddingsService):
        """Новый текст вызывает промах, старый — попадание."""
        await service_st.embed_texts(["старый текст"])
        stats_before = service_st.cache_stats()

        await service_st.embed_texts(["старый текст", "новый текст"])
        stats_after = service_st.cache_stats()

        # Должен быть 1 hit (старый) и 1 новый miss
        assert stats_after["hits"] == stats_before["hits"] + 1
        assert stats_after["misses"] == stats_before["misses"] + 1


class TestE5Methods:
    """Проверка E5-специфичных методов (если модель E5)."""

    @pytest.mark.asyncio
    async def test_non_e5_raises_not_implemented(self):
        """Не-E5 модель выбрасывает NotImplementedError."""
        svc = create_embeddings_service(
            provider="sentence_transformers",
            model="all-MiniLM-L6-v2",
        )
        with pytest.raises(NotImplementedError):
            await svc.embed_query("запрос")
        with pytest.raises(NotImplementedError):
            await svc.embed_documents(["документ"])

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires E5 model download (~1GB); run manually")
    async def test_e5_prefix_improves_score(self):
        """Smoke-тест: с префиксом query:/passage: score выше, чем без."""
        svc = create_embeddings_service(
            provider="sentence_transformers",
            model="intfloat/multilingual-e5-large",
        )
        query = "Как вернуть товар?"
        doc = "Возврат товара возможен в течение 14 дней с момента покупки."

        # С префиксами
        q_vec = await svc.embed_query(query)
        d_vecs = await svc.embed_documents([doc])
        score_with = _cosine(q_vec, d_vecs[0])

        # Без префиксов
        q_vec_raw = await svc.embed_one(query)
        d_vec_raw = await svc.embed_one(doc)
        score_without = _cosine(q_vec_raw, d_vec_raw)

        print(f"Score with prefixes: {score_with:.4f}, without: {score_without:.4f}")
        # Префиксы должны давать заметно лучший результат
        assert score_with > score_without, (
            f"Expected prefix score ({score_with:.4f}) > raw score ({score_without:.4f})"
        )


class TestMiniBenchmark:
    """Проверка качества на mini-benchmark из домена.

    ВАЖНО: all-MiniLM-L6-v2 — английская модель (не различает русские тексты).
    Для русского домена нужна multilingual-модель (intfloat/multilingual-e5-large).
    Тесты в этом классе требуют MULTILINGUAL_MODEL=1 в окружении.
    """

    @pytest.fixture
    async def multilingual_service(self) -> EmbeddingsService:
        """Сервис с мультиязычной моделью. Требует загрузки ~420 MB."""
        import os
        if not os.environ.get("MULTILINGUAL_MODEL"):
            pytest.skip("Set MULTILINGUAL_MODEL=1 to run benchmark tests "
                        "(requires paraphrase-multilingual-MiniLM-L12-v2)")
        return create_embeddings_service(
            provider="sentence_transformers",
            model="paraphrase-multilingual-MiniLM-L12-v2",
        )

    @pytest.mark.asyncio
    async def test_relevant_scores_higher_than_irrelevant(
        self, multilingual_service: EmbeddingsService, mini_benchmark: list[dict]
    ):
        """Для каждой пары: cosine(query, relevant) > cosine(query, irrelevant)."""
        svc = multilingual_service  # type: ignore[assignment]
        for i, pair in enumerate(mini_benchmark):
            query = pair["query"]
            relevant = pair["relevant"]
            irrelevant = pair["irrelevant"]

            vecs = await svc.embed_texts([query, relevant, irrelevant])
            q_vec, rel_vec, irr_vec = vecs[0], vecs[1], vecs[2]

            score_rel = _cosine(q_vec, rel_vec)
            score_irr = _cosine(q_vec, irr_vec)

            assert score_rel > score_irr, (
                f"Pair {i}: rel={score_rel:.4f} <= irr={score_irr:.4f}\n"
                f"  query: {query[:80]}...\n"
                f"  relevant: {relevant[:80]}...\n"
                f"  irrelevant: {irrelevant[:80]}..."
            )

    @pytest.mark.asyncio
    async def test_threshold_separation(
        self, multilingual_service: EmbeddingsService, mini_benchmark: list[dict]
    ):
        """Разрыв между минимальным релевантным и максимальным нерелевантным > 0."""
        svc = multilingual_service  # type: ignore[assignment]
        rel_scores = []
        irr_scores = []

        for pair in mini_benchmark:
            vecs = await svc.embed_texts(
                [pair["query"], pair["relevant"], pair["irrelevant"]]
            )
            rel_scores.append(_cosine(vecs[0], vecs[1]))
            irr_scores.append(_cosine(vecs[0], vecs[2]))

        min_rel = min(rel_scores)
        max_irr = max(irr_scores)
        gap = min_rel - max_irr

        print(f"Min relevant: {min_rel:.4f}, Max irrelevant: {max_irr:.4f}, Gap: {gap:.4f}")
        assert gap > 0.0, (
            f"No separation: min_rel={min_rel:.4f} <= max_irr={max_irr:.4f}"
        )


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    """Косинусная близость между двумя векторами."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)
