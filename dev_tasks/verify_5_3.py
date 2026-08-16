"""CLI-скрипт для проверки критериев самопроверки ДЗ 5.3 (RAG с LlamaIndex).

Критерии:
1. app/services/rag.py возвращает dict с answer + >=3 элементами в sources.
2. app/services/rag_baremetal.py — совместимый по форме результат (те же Qdrant/embed).
3. docs/rag.md содержит: версии, решение по коллекции, таблицу «LlamaIndex vs bare-metal»,
   прогон 5 вопросов (3 хороших / 1 средний / 1 вне базы с гипотезами).
4. POST /rag/query возвращает JSON с answer и sources; индекс не пересоздаётся на запрос.
5. Конфигурация (LLM, embed, chunk_size, chunk_overlap, top_k, коллекции) — в config/.env,
   без хардкода в сервисах.
6. Вне-базы вопрос не выдумывает ответ (честный fallback от LLM либо отсечение по top_score).

Использование:
    uv run python dev_tasks/verify_5_3.py   # требуется живой Qdrant + Ollama + E5
"""

import asyncio
import sys
from pathlib import Path

# Windows: консоль может быть в cp1251 — переключаем stdout на UTF-8,
# чтобы print с кириллицей/стрелками не падал с UnicodeEncodeError.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from app.core.config import get_settings

GOOD_QUESTION = "Что входит в состав КПС «Тарифы — Реестр ОИС»?"
OUT_QUESTION = "Какая завтра погода в Москве?"
FALLBACK_MARKERS = ("нет информации", "не нашёл", "не нашла", "нет ответа", "не знаю")


def _section(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def check_config() -> None:
    """Критерий 5: конфигурация без хардкода в сервисах."""
    _section("Критерий 5: конфигурация (config.py / .env.example, без хардкода)")

    cfg = Path("app/core/config.py").read_text(encoding="utf-8")
    for field in (
        "rag_data_dir", "rag_collection", "rag_collection_bare", "rag_llm_model",
        "rag_top_k", "rag_chunk_size", "rag_chunk_overlap", "rag_score_threshold",
    ):
        assert field in cfg, f"В config.py нет поля {field}"
    print("  [OK] config.py содержит все RAG-поля")

    env = Path(".env.example").read_text(encoding="utf-8")
    for var in (
        "RAG_DATA_DIR", "RAG_COLLECTION", "RAG_COLLECTION_BARE", "RAG_LLM_MODEL",
        "RAG_TOP_K", "RAG_CHUNK_SIZE", "RAG_CHUNK_OVERLAP", "RAG_SCORE_THRESHOLD",
    ):
        assert var in env, f"В .env.example нет {var}"
    print("  [OK] .env.example содержит все RAG-переменные")

    rag_src = Path("app/services/rag.py").read_text(encoding="utf-8")
    bm_src = Path("app/services/rag_baremetal.py").read_text(encoding="utf-8")
    # Модели/URL/имена коллекций не захардкожены в сервисах — только через settings.
    for src, name in ((rag_src, "rag.py"), (bm_src, "rag_baremetal.py")):
        assert "gemma3:4b" not in src, f"Хардкод модели в {name}"
        assert "rag_block_03" not in src, f"Хардкод имени коллекции в {name}"
        assert "localhost:6333" not in src, f"Хардкод URL Qdrant в {name}"
    print("  [OK] в сервисах нет хардкода модели/коллекции/URL")

    print()
    print("[OK] Критерий 5 ВЫПОЛНЕН: конфигурация вынесена в config/.env")


def check_docs() -> None:
    """Критерии 3 и 7: содержимое docs/rag.md."""
    _section("Критерии 3+7: docs/rag.md")

    doc = Path("docs/rag.md")
    assert doc.exists(), "docs/rag.md не найден"
    content = doc.read_text(encoding="utf-8")

    # Версии зависимостей
    for pkg in ("llama-index", "llama-index-vector-stores-qdrant",
                "llama-index-embeddings-huggingface", "llama-index-readers-file"):
        assert pkg in content, f"Нет версии {pkg} в docs/rag.md"
    print("  [OK] зафиксированы версии зависимостей")

    # Решение по коллекции
    assert "Решение по коллекции" in content, "Нет раздела 'Решение по коллекции'"
    assert "rag_block_03" in content and "rag_block_03_bare" in content
    print("  [OK] решение по коллекции описано")

    # Таблица сравнения
    assert "LlamaIndex vs bare-metal" in content, "Нет таблицы сравнения"
    for row in ("Строк кода", "Поддержка форматов", "re-ranker"):
        assert row in content, f"Нет строки таблицы '{row}'"
    print("  [OK] таблица 'LlamaIndex vs bare-metal' присутствует")

    # Прогон 5 вопросов
    assert "Прогон 5 вопросов" in content, "Нет раздела 'Прогон 5 вопросов'"
    for label in ("хороший", "средний", "вне базы"):
        assert label in content, f"Нет типа '{label}' в прогоне"
    assert "Гипотеза" in content, "Нет колонки 'Гипотеза'"
    print("  [OK] прогон 5 вопросов (3 хороших / 1 средний / 1 вне базы) с гипотезами")

    print()
    print("[OK] Критерии 3+7 ВЫПОЛНЕНЫ: docs/rag.md полный")


async def _check_rag_contract(settings) -> None:
    """Критерий 1: rag.py — dict с answer + >=3 sources."""
    _section("Критерий 1: RAGService (LlamaIndex) — контракт answer()")

    from app.services.rag import RAGService

    svc = RAGService(settings)
    svc.build()
    result = await svc.answer(GOOD_QUESTION)
    await svc.close()

    assert set(result) == {"answer", "top_score", "sources"}, f"Неверные ключи: {result.keys()}"
    assert isinstance(result["answer"], str) and result["answer"], "Пустой answer"
    assert len(result["sources"]) >= 3, f"sources < 3: {len(result['sources'])}"
    for s in result["sources"]:
        assert set(s) == {"text", "source", "score"}, f"Неверные ключи source: {s.keys()}"
        assert s["source"], "Пустой source в sources"
    print(f"  [OK] answer непустой, sources={len(result['sources'])}, top_score={result['top_score']}")
    print()
    print("[OK] Критерий 1 ВЫПОЛНЕН: rag.py возвращает answer + >=3 sources")


async def _check_baremetal_contract(settings) -> None:
    """Критерий 2: rag_baremetal.py — совместимый результат."""
    _section("Критерий 2: BareMetalRAG — совместимый контракт")

    from app.services.rag_baremetal import BareMetalRAG

    svc = BareMetalRAG(settings)
    await svc.ensure_indexed()
    result = await svc.answer(GOOD_QUESTION)
    await svc.close()

    assert set(result) == {"answer", "top_score", "sources"}, f"Неверные ключи: {result.keys()}"
    assert result["answer"], "Пустой answer"
    assert len(result["sources"]) >= 3, f"sources < 3: {len(result['sources'])}"
    for s in result["sources"]:
        assert set(s) == {"text", "source", "score"}
    print(f"  [OK] совместимый результат, top_score={result['top_score']}, "
          f"top-1={result['sources'][0]['source']}")
    print()
    print("[OK] Критерий 2 ВЫПОЛНЕН: bare-metal возвращает совместимый по форме результат")


async def _check_fallback(settings) -> None:
    """Критерий 6: вне-базы вопрос не выдумывает ответ."""
    _section("Критерий 6: fallback на вне-базы вопросе")

    from app.services.rag import RAGService

    svc = RAGService(settings)
    svc.build()
    result = await svc.answer(OUT_QUESTION)
    await svc.close()

    answer = result["answer"].lower()
    fallback_by_llm = any(m in answer for m in FALLBACK_MARKERS)
    fallback_by_score = result["top_score"] < settings.rag_score_threshold
    print(f"  question: {OUT_QUESTION}")
    print(f"  answer:   {result['answer'][:80]}")
    print(f"  top_score={result['top_score']} (порог {settings.rag_score_threshold})")
    assert fallback_by_llm or fallback_by_score, (
        "Система выдумала ответ на вне-базы вопрос"
    )
    print(f"  [OK] fallback: {'LLM честно отказал' if fallback_by_llm else ''}"
          f"{' + ' if fallback_by_llm and fallback_by_score else ''}"
          f"{'отсечение по top_score' if fallback_by_score else ''}")
    print()
    print("[OK] Критерий 6 ВЫПОЛНЕН: вне-базы вопрос не выдумывает ответ")


async def _check_endpoint(settings) -> None:
    """Критерий 4: POST /rag/query возвращает JSON, индекс не пересоздаётся."""
    _section("Критерий 4: POST /rag/query")

    from httpx import ASGITransport, AsyncClient

    from app.main import app
    from app.services.rag import RAGService

    svc = RAGService(settings)
    svc.build()
    app.state.rag_service = svc

    from qdrant_client import AsyncQdrantClient
    qdrant_key = (
        settings.qdrant_api_key.get_secret_value()
        if settings.qdrant_api_key is not None else None
    )
    qc = AsyncQdrantClient(url=settings.qdrant_url, api_key=qdrant_key)
    before = (await qc.get_collection(settings.rag_collection)).points_count or 0

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r1 = await client.post("/rag/query", json={"question": GOOD_QUESTION})
        r2 = await client.post("/rag/query", json={"question": GOOD_QUESTION})

    assert r1.status_code == 200, f"Статус {r1.status_code}: {r1.text[:200]}"
    data = r1.json()
    assert data["answer"], "Пустой answer в ответе endpoint"
    assert len(data["sources"]) >= 3, "sources < 3 в ответе endpoint"
    assert data["top_score"] is not None
    print(f"  [OK] /rag/query → 200, answer + {len(data['sources'])} sources")

    after = (await qc.get_collection(settings.rag_collection)).points_count or 0
    await qc.close()
    assert before == after, f"Индекс пересоздан: было {before}, стало {after}"
    print(f"  [OK] индекс не пересоздаётся (точек до/после: {before}/{after})")

    await svc.close()
    print()
    print("[OK] Критерий 4 ВЫПОЛНЕН: /rag/query работает, индекс инициализируется один раз")


async def main() -> None:
    print()
    print("=" * 60)
    print("САМОПРОВЕРКА ДЗ 5.3 — Архитектура RAG с LlamaIndex")
    print("=" * 60)

    settings = get_settings()

    check_config()
    check_docs()
    await _check_rag_contract(settings)
    await _check_baremetal_contract(settings)
    await _check_fallback(settings)
    await _check_endpoint(settings)

    print()
    print("=" * 60)
    print("ИТОГ САМОПРОВЕРКИ")
    print("=" * 60)
    print("  [OK] Критерий 1: rag.py — answer + >=3 sources")
    print("  [OK] Критерий 2: rag_baremetal.py — совместимый результат")
    print("  [OK] Критерий 3: docs/rag.md — версии, коллекция, таблица, прогон")
    print("  [OK] Критерий 4: POST /rag/query — JSON, индекс один раз")
    print("  [OK] Критерий 5: конфигурация в config/.env, без хардкода")
    print("  [OK] Критерий 6: вне-базы вопрос не выдумывает ответ")
    print()


if __name__ == "__main__":
    asyncio.run(main())
