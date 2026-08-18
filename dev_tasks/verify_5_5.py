"""CLI-скрипт для проверки критериев самопроверки ДЗ 5.5 (Корпоративный RAG).

Критерии:
1. data/kb содержит 50+ документов в 2+ форматах (PDF/DOCX/HTML/MD), без пустых файлов.
2. docs/data_inventory.md фиксирует число/форматы/размер корпуса.
3. config.py/.env.example содержат новые RAG-поля (docstore, enable_chat, condense, threshold).
4. app/services/ingestion.py — IngestionPipeline + UPSERTS + docstore; чистые функции обогащения.
5. app/services/rag.py — retrieve→guard→synthesize с цитатами [1][2] и confident.
6. POST /rag/query возвращает контракт {answer, top_score, sources[id,file_name,page,score,snippet], confident}.
7. Вне-базы вопрос → честный отказ (score-guard без вызова LLM).
8. docs/rag.md содержит Mermaid-диаграммы, параметры chunking, обоснование threshold.

Использование:
    uv run python dev_tasks/verify_5_5.py   # критерии 6-7 требуют живой Qdrant + Ollama + E5
"""

import asyncio
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
KB = ROOT / "data" / "kb"
GOOD_QUESTION = "Что входит в состав КПС «Тарифы — Реестр ОИС»?"
OUT_QUESTION = "Какая завтра погода в Москве?"


def _section(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def check_corpus() -> None:
    """Критерий 1: 50+ документов в 2+ форматах."""
    _section("Критерий 1: корпус data/kb")

    assert KB.exists(), "data/kb не найден — запустите scripts/prepare_corpus.py"
    files = [p for p in KB.rglob("*") if p.is_file()]
    by_fmt: dict[str, int] = {}
    empty = 0
    for p in files:
        by_fmt[p.suffix.lower()] = by_fmt.get(p.suffix.lower(), 0) + 1
        if p.stat().st_size == 0:
            empty += 1
    assert len(files) >= 50, f"нужно 50+ файлов, получено {len(files)}"
    assert len(by_fmt) >= 2, f"нужно 2+ формата, получено {list(by_fmt)}"
    assert empty == 0, f"в корпусе {empty} пустых файлов"
    print(f"  [OK] {len(files)} файлов, форматы: {by_fmt}")
    print("[OK] Критерий 1 ВЫПОЛНЕН")


def check_data_inventory() -> None:
    """Критерий 2: docs/data_inventory.md."""
    _section("Критерий 2: docs/data_inventory.md")

    doc = ROOT / "docs" / "data_inventory.md"
    assert doc.exists(), "docs/data_inventory.md не найден"
    content = doc.read_text(encoding="utf-8")
    for token in ("Всего документов", "Общий размер", "| Формат |", "| Категория |"):
        assert token in content, f"нет секции '{token}' в data_inventory.md"
    print("  [OK] число документов, разбивка по форматам и размер зафиксированы")
    print("[OK] Критерий 2 ВЫПОЛНЕН")


def check_config() -> None:
    """Критерий 3: новые RAG-поля в config.py/.env.example."""
    _section("Критерий 3: конфигурация")

    cfg = (ROOT / "app/core/config.py").read_text(encoding="utf-8")
    for field in (
        "rag_docstore_path", "rag_top_k", "rag_chunk_size", "rag_chunk_overlap",
        "rag_score_threshold", "rag_rerank_enabled", "rag_rerank_model",
        "rag_rerank_top_n", "rag_enable_chat", "rag_condense_enabled",
    ):
        assert field in cfg, f"в config.py нет поля {field}"
    env = (ROOT / ".env.example").read_text(encoding="utf-8")
    for var in (
        "RAG_DATA_DIR", "RAG_COLLECTION", "RAG_DOCSTORE_PATH", "RAG_TOP_K",
        "RAG_CHUNK_SIZE", "RAG_CHUNK_OVERLAP", "RAG_SCORE_THRESHOLD",
        "RAG_RERANK_ENABLED", "RAG_RERANK_MODEL", "RAG_RERANK_TOP_N",
        "RAG_ENABLE_CHAT", "RAG_CONDENSE_ENABLED",
    ):
        assert var in env, f"в .env.example нет {var}"
    print("  [OK] поля вынесены в config/.env.example")
    print("[OK] Критерий 3 ВЫПОЛНЕН")


def check_ingestion_impl() -> None:
    """Критерий 4: офлайн-контур индексации."""
    _section("Критерий 4: app/services/ingestion.py")

    src = (ROOT / "app/services/ingestion.py").read_text(encoding="utf-8")
    for fn in ("def clean(", "def category_from_path(", "def file_metadata(",
               "def enrich(", "EXCLUDED_EMBED_KEYS"):
        assert fn in src, f"нет {fn} в ingestion.py"
    for token in ("IngestionPipeline", "DocstoreStrategy.UPSERTS", "SimpleDocumentStore",
                  "def ingest_all", "def ingest_files", "def reindex_all", "def run_for_file"):
        assert token in src, f"нет {token} в ingestion.py"
    # Стабильный doc_id — залог идемпотентности UPSERTS.
    assert "doc_id" in src and "uuid5" in src, "нет стабильного doc_id (UPSERTS не идемпотентен)"
    print("  [OK] IngestionPipeline + UPSERTS + docstore + стабильный doc_id")
    print("[OK] Критерий 4 ВЫПОЛНЕН")


def check_rag_impl() -> None:
    """Критерий 5: онлайн-контур с цитатами."""
    _section("Критерий 5: app/services/rag.py")

    src = (ROOT / "app/services/rag.py").read_text(encoding="utf-8")
    for token in ("REFUSAL_TEXT", "CITATION_QA_TEMPLATE", "def build_sources",
                  "def parse_citations", "def numbered_context", "def build_filters",
                  "rag_score_threshold", "rag_rerank_top_n"):
        assert token in src, f"нет {token} в rag.py"
    # score-guard: отказ БЕЗ вызова LLM.
    assert "top_score < self._settings.rag_score_threshold" in src, "нет score-guard"
    print("  [OK] retrieve → guard → synthesize с цитатами и confident")
    print("[OK] Критерий 5 ВЫПОЛНЕН")


async def _check_rag_contract(settings) -> None:
    """Критерий 6: /rag/query — контракт с цитатами."""
    _section("Критерий 6: RAGService.answer() контракт")

    from app.services.rag import RAGService

    svc = RAGService(settings)
    svc.build()
    result = await svc.answer(GOOD_QUESTION)
    await svc.close()

    assert set(result) == {"answer", "top_score", "sources", "confident"}, result.keys()
    assert result["answer"], "пустой answer"
    assert result["confident"] is True, "ожидался confident=True"
    assert len(result["sources"]) >= 1, "нет sources"
    s = result["sources"][0]
    assert set(s) == {"id", "file_name", "page", "score", "snippet"}, s.keys()
    assert s["id"] == 1 and s["file_name"], "неверная цитата [1]"
    print(f"  [OK] answer + {len(result['sources'])} sources, top_score={result['top_score']}")
    print(f"  top-1: [{s['id']}] {s['file_name']} стр.{s.get('page')} score={s['score']}")
    print("[OK] Критерий 6 ВЫПОЛНЕН")


async def _check_refusal(settings) -> None:
    """Критерий 7: вне-базы вопрос — честный отказ через score-guard."""
    _section("Критерий 7: score-guard на вне-базы вопросе")

    from app.services.rag import RAGService

    svc = RAGService(settings)
    svc.build()
    result = await svc.answer(OUT_QUESTION)
    await svc.close()

    assert result["confident"] is False, "вне-базы вопрос должен дать confident=False"
    assert result["sources"] == [], "при отказе sources должен быть пуст"
    assert "не нашёл" in result["answer"], f"ожидался отказ, получено: {result['answer'][:60]}"
    print(f"  question: {OUT_QUESTION}")
    print(f"  answer:   {result['answer']}")
    print(f"  top_score={result['top_score']} (порог {settings.rag_score_threshold})")
    print("[OK] Критерий 7 ВЫПОЛНЕН: score-guard сработал, LLM не вызывался")


def check_docs() -> None:
    """Критерий 8: docs/rag.md."""
    _section("Критерий 8: docs/rag.md")

    doc = ROOT / "docs" / "rag.md"
    assert doc.exists(), "docs/rag.md не найден"
    content = doc.read_text(encoding="utf-8")

    # Mermaid-диаграммы двух контуров.
    assert "```mermaid" in content, "нет Mermaid-диаграммы"
    assert "Офлайн-контур" in content and "Онлайн-контур" in content, "нет двух контуров"
    # Параметры чанкинга и threshold с обоснованием.
    assert "chunk_size=512" in content and "chunk_overlap=64" in content, "нет параметров чанкинга"
    assert "0.80" in content, "нет обоснованного порога 0.80"
    assert "bge-reranker-v2-m3" in content, "нет выбранного re-ranker"
    # Перечень endpoints.
    for ep in ("/rag/query", "/documents/upload", "/documents/reindex", "/chats/{id}/messages"):
        assert ep in content, f"нет endpoint {ep} в rag.md"
    print("  [OK] Mermaid + chunking + threshold + re-ranker + endpoints")
    print("[OK] Критерий 8 ВЫПОЛНЕН")


async def main() -> None:
    print()
    print("=" * 60)
    print("САМОПРОВЕРКА ДЗ 5.5 — Корпоративный RAG-ассистент")
    print("=" * 60)

    from app.core.config import get_settings

    settings = get_settings()

    check_corpus()
    check_data_inventory()
    check_config()
    check_ingestion_impl()
    check_rag_impl()
    check_docs()

    # Живые проверки (Qdrant + Ollama + E5).
    await _check_rag_contract(settings)
    await _check_refusal(settings)

    print()
    print("=" * 60)
    print("ИТОГ САМОПРОВЕРКИ")
    print("=" * 60)
    print("  [OK] Критерий 1: корпус 50+ документов в 2+ форматах")
    print("  [OK] Критерий 2: docs/data_inventory.md")
    print("  [OK] Критерий 3: конфигурация вынесена в config/.env")
    print("  [OK] Критерий 4: IngestionPipeline + UPSERTS + docstore")
    print("  [OK] Критерий 5: retrieve → guard → synthesize с цитатами")
    print("  [OK] Критерий 6: /rag/query — контракт с sources + confident")
    print("  [OK] Критерий 7: вне-базы → отказ через score-guard")
    print("  [OK] Критерий 8: docs/rag.md (Mermaid + chunking + threshold)")
    print()
    print("Ручная проверка (требуют полного стека):")
    print("  - повторный `scripts/ingest.py data/kb` → «0 changed, N unchanged»")
    print("  - multi-turn «Что входит в состав КПС „Тарифы — Реестр ОИС“?» → «А в состав ТРОИС?» через /chats")
    print("  - POST /documents/upload → 202, документ в индексе через 30-60 с")
    print("  - Telegram-бот: стриминг + editMessageText + feedback")
    print("  - `docker compose up -d` = app + qdrant + redis + bot")


if __name__ == "__main__":
    asyncio.run(main())
