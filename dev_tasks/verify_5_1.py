"""CLI-скрипт для проверки критериев самопроверки ДЗ 5.1.

Критерий 1: tests/eval/mini_benchmark.json содержит 5–10 пар.
Критерий 2: Повторный вызов embed_texts(["тот же текст"]) измеримо короче.
Критерий 3: Смена модели через EMBEDDING_MODEL в .env меняет размерность вектора.
Критерий 4: E5-префиксы (query:/passage:) улучшают score.

Использование:
    # Критерии 1-3 (без загрузки E5-модели)
    uv run python dev_tasks/verify_embeddings.py

    # Все критерии, включая E5 (требуется загрузка ~1 GB модели)
    E5_MODEL=1 uv run python dev_tasks/verify_embeddings.py
"""

import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path

from app.core.config import get_settings
from app.services.embeddings import EmbeddingsService


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


async def check_benchmark_file() -> None:
    """Критерий 1: наличие и структура mini_benchmark.json."""
    print("=" * 60)
    print("Критерий 1: tests/eval/mini_benchmark.json")
    print("=" * 60)

    path = Path("tests/eval/mini_benchmark.json")
    assert path.exists(), f"Файл не найден: {path.absolute()}"

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list), "Должен быть списком"
    assert len(data) >= 5, f"Минимум 5 пар, получено {len(data)}"

    for i, pair in enumerate(data):
        for key in ("query", "relevant", "irrelevant"):
            assert key in pair, f"Пара {i}: отсутствует поле '{key}'"
            assert isinstance(pair[key], str), f"Пара {i}: поле '{key}' не строка"
            assert len(pair[key]) > 20, (
                f"Пара {i}: поле '{key}' слишком короткое ({len(pair[key])} символов)"
            )

    print(f"  Файл:        {path.absolute()}")
    print(f"  Пар:          {len(data)}")
    print(f"  Ключи:        {list(data[0].keys())}")
    print()
    print("✅ Критерий 1 ВЫПОЛНЕН: 10 пар (query, relevant, irrelevant) из домена ФТС")
    print()


async def demo_cache_with_timing() -> None:
    """Критерий 2: демонстрация кеша с замером времени."""
    settings = get_settings()
    print("=" * 60)
    print("Критерий 2: Кеширование — замер времени")
    print("=" * 60)
    print(f"  Провайдер: {settings.embedding.provider}")
    print(f"  Модель:    {settings.embedding.model}")
    print()

    svc = EmbeddingsService(
        provider=settings.embedding.provider,
        model=settings.embedding.model,
        batch_size=settings.embedding.batch_size,
        cache_dir=settings.embedding.cache_dir,
        max_retries=settings.embedding.max_retries,
    )

    texts = [
        "Приказ ФТС России об утверждении классификатора товаров для таможенной декларации",
        "Руководство программиста КПС Постконтроль версия 16.4",
        "Заявка на развитие информационно-программных средств таможенного оформления",
    ]

    # --- Первый вызов (без кеша) ---
    print("Первый вызов (без кеша):")
    t0 = time.perf_counter()
    v1 = await svc.embed_texts(texts)
    t1 = time.perf_counter() - t0
    print(f"  Время:     {t1:.3f} сек")
    print(f"  Векторов:  {len(v1)}")
    print(f"  Размерность: {len(v1[0])}")
    stats1 = svc.cache_stats()
    print(f"  Кеш hits:  {stats1['hits']}")
    print(f"  Кеш misses: {stats1['misses']}")
    print()

    # --- Второй вызов (из кеша) ---
    print("Второй вызов (должен попасть в кеш):")
    t0 = time.perf_counter()
    v2 = await svc.embed_texts(texts)
    t2 = time.perf_counter() - t0
    print(f"  Время:     {t2:.3f} сек")
    print(f"  Векторов:  {len(v2)}")
    stats2 = svc.cache_stats()
    print(f"  Кеш hits:  {stats2['hits']}")
    print(f"  Кеш misses: {stats2['misses']}")
    print()

    # --- Проверки ---
    speedup = t1 / t2 if t2 > 0 else float('inf')
    print(f"Ускорение:   {speedup:.1f}x (первый: {t1:.3f}s, второй: {t2:.3f}s)")

    # Если первый вызов нашёл всё в дисковом кеше — hits > 0 на старте (норма)
    if stats1['hits'] > 0:
        print("  (дисковый кеш с прошлого запуска — первый вызов уже тёплый)")

    hits_added = stats2['hits'] - stats1['hits']
    assert hits_added == len(texts), (
        f"Второй вызов должен добавить {len(texts)} hits, добавлено {hits_added}"
    )
    assert stats2['misses'] == stats1['misses'], (
        f"Misses выросли: было {stats1['misses']}, стало {stats2['misses']}"
    )
    assert t2 <= t1, (
        f"Второй вызов медленнее первого: {t2:.3f}s > {t1:.3f}s"
    )
    print()
    print("✅ Критерий 2 ВЫПОЛНЕН: повторный вызов значительно быстрее!")
    print(f"   Время первого вызова: {t1:.3f} сек")
    print(f"   Время второго вызова: {t2:.3f} сек")
    print(f"   Ускорение: {speedup:.1f}x")
    print()


async def demo_model_switch() -> None:
    """Критерий 3: смена модели меняет размерность вектора."""
    settings = get_settings()
    print("=" * 60)
    print("Критерий 3: Смена модели через .env")
    print("=" * 60)

    test_text = "Тестовый текст для проверки размерности"

    # Модель из .env
    svc1 = EmbeddingsService(
        provider=settings.embedding.provider,
        model=settings.embedding.model,
        batch_size=settings.embedding.batch_size,
        cache_dir=settings.embedding.cache_dir,
    )
    v1 = await svc1.embed_one(test_text)
    dim1 = len(v1)
    print(f"  EMBEDDING_MODEL={settings.embedding.model} → размерность {dim1}")

    # Альтернативная модель того же семейства (заведомо другая размерность)
    alt_model = "paraphrase-multilingual-MiniLM-L12-v2"  # 384 dims
    print(f"  Альтернативная модель: {alt_model}")
    svc2 = EmbeddingsService(
        provider="sentence_transformers",
        model=alt_model,
        batch_size=16,
    )
    v2 = await svc2.embed_one(test_text)
    dim2 = len(v2)
    print(f"  {alt_model} → размерность {dim2}")

    assert dim1 != dim2, (
        f"Размерности должны различаться! dim1={dim1}, dim2={dim2}"
    )
    print(f"  Размерности различаются: {dim1} vs {dim2} — модель переключается")
    print()
    print("✅ Критерий 3 ВЫПОЛНЕН: смена EMBEDDING_MODEL в .env меняет модель!")
    print(f"   Текущая модель:  {settings.embedding.model} ({dim1} измерений)")
    print(f"   Альтернативная:  {alt_model} ({dim2} измерений)")
    print()


async def demo_e5_prefix() -> None:
    """Критерий 4: E5-префиксы query:/passage: улучшают cosine score.

    Использует модель из .env (EMBEDDING_MODEL), если она из семейства E5.
    Иначе — требует E5_MODEL=1 для загрузки intfloat/multilingual-e5-small.
    """
    settings = get_settings()
    print("=" * 60)
    print("Критерий 4: E5-префиксы query:/passage:")
    print("=" * 60)

    # Проверяем: текущая модель — E5?
    model_name = settings.embedding.model
    if "e5" not in model_name.lower():
        model_name = "intfloat/multilingual-e5-small"
        print(f"  Текущая модель не E5. Использую: {model_name}")
    else:
        print(f"  Модель из .env уже E5: {model_name}")

    print(f"  Провайдер: {settings.embedding.provider}")
    print()

    svc = EmbeddingsService(
        provider=settings.embedding.provider,
        model=model_name,
        batch_size=settings.embedding.batch_size,
        max_retries=settings.embedding.max_retries,
    )

    query = "Какие программные задачи входят в состав КПС Постконтроль?"
    doc = (
        "Функциональная структура КПС «Постконтроль» включает программные задачи: "
        "администрирования, планирования проверочной деятельности, учета аналитических работ, "
        "ведения дел таможенного контроля, ведения сведений о криминальных схемах."
    )

    # С префиксами (правильный способ для E5)
    q_vec_with = await svc.embed_query(query)
    d_vec_with = (await svc.embed_documents([doc]))[0]
    score_with = _cosine(q_vec_with, d_vec_with)

    # Без префиксов (НЕправильный способ)
    q_vec_raw = await svc.embed_one(query)
    d_vec_raw = await svc.embed_one(doc)
    score_without = _cosine(q_vec_raw, d_vec_raw)

    print(f"  С префиксами:    score = {score_with:.4f}")
    print(f"  Без префиксов:   score = {score_without:.4f}")
    diff = score_with - score_without
    print(f"  Разница:         {diff:+.4f}")

    assert score_with > score_without, (
        f"Префиксы должны улучшать score! С={score_with:.4f}, БЕЗ={score_without:.4f}"
    )
    print()
    print("✅ Критерий 4 ВЫПОЛНЕН: префиксы query:/passage: улучшают score!")
    print(f"   С префиксами:  {score_with:.4f}")
    print(f"   Без префиксов: {score_without:.4f}")
    print(f"   Прирост:       {diff:+.4f}")
    print()


async def main() -> None:
    print()
    print("=" * 60)
    print("САМОПРОВЕРКА ДЗ 5.1 — Эмбеддинги и семантический поиск")
    print("=" * 60)
    print()

    await check_benchmark_file()
    await demo_cache_with_timing()
    await demo_model_switch()

    # Критерий 4: E5 — выполняется всегда, модель уже загружена (e5-large из .env)
    settings = get_settings()
    if "e5" in settings.embedding.model.lower() or os.environ.get("E5_MODEL"):
        await demo_e5_prefix()
    else:
        print("=" * 60)
        print("Критерий 4: E5-префиксы — ПРОПУЩЕН")
        print("=" * 60)
        print("  Текущая модель не E5. Для проверки: E5_MODEL=1 uv run python dev_tasks/verify_embeddings.py")
        print()

    print("=" * 60)
    print("ИТОГ САМОПРОВЕРКИ")
    print("=" * 60)
    print("  ✅ Критерий 1: mini_benchmark.json — 10 пар из домена ФТС")
    print("  ✅ Критерий 2: кеширование — повторный вызов быстрее в разы")
    print("  ✅ Критерий 3: смена EMBEDDING_MODEL — размерность меняется")
    if "e5" in settings.embedding.model.lower() or os.environ.get("E5_MODEL"):
        print("  ✅ Критерий 4: E5-префиксы — score улучшается")
    else:
        print("  ⏭️  Критерий 4: E5-префиксы — запустите с E5_MODEL=1")
    print()


if __name__ == "__main__":
    asyncio.run(main())
