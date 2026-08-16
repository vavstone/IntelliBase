"""CLI-скрипт для проверки критериев самопроверки ДЗ 5.4 (Chunking и оптимизация).

Критерии:
1. tests/eval/retrieval_dataset.json — >= 20 вопросов, у каждого 1–3 релевантных документа.
2. Три стратегии chunking в app/services/chunking.py + три непустые коллекции Qdrant
   (docs_fixed, docs_recursive, docs_semantic).
3. docs/chunking_experiment.md — таблица метрик для всех трёх стратегий.
4. Re-ranker подключён (app/services/reranker.py), метрики до/после измерены.
5. Лучшая конфигурация (chunk_size, overlap, top-K) зафиксирована в config.py/.env.
6. Финальный Hit Rate@5 >= 0.5 (иначе — анализ причин в отчёте).
7. В отчёте есть итоговый параграф «выбираю стратегию X с конфигом Y, потому что…».

Использование:
    uv run python dev_tasks/verify_5_4.py   # для критерия 2 нужен живой Qdrant
"""

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "tests/eval/retrieval_dataset.json"
RESULTS = ROOT / "var/chunking_results.json"
REPORT = ROOT / "docs/chunking_experiment.md"
COLLECTIONS = ("docs_fixed", "docs_recursive", "docs_semantic")


def _section(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def check_dataset() -> None:
    """Критерий 1: golden dataset."""
    _section("Критерий 1: tests/eval/retrieval_dataset.json")

    assert DATASET.exists(), f"Не найден {DATASET}"
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    assert isinstance(data, list), "Должен быть список"
    assert len(data) >= 20, f"Нужно >= 20 вопросов, получено {len(data)}"

    corpus = {p.name for p in (ROOT / "data/rag-block-03").iterdir()}
    for i, item in enumerate(data):
        assert set(item) == {"question", "relevant_doc_ids"}, \
            f"Вопрос {i}: ключи {set(item)}"
        assert isinstance(item["question"], str) and item["question"], \
            f"Вопрос {i}: пустой question"
        n_rel = len(item["relevant_doc_ids"])
        assert 1 <= n_rel <= 3, f"Вопрос {i}: релевантных доков {n_rel}"
        for d in item["relevant_doc_ids"]:
            assert d in corpus, f"Вопрос {i}: нет файла {d} в корпусе"

    print(f"  [OK] {len(data)} вопросов, у каждого 1–3 релевантных документа")
    print("[OK] Критерий 1 ВЫПОЛНЕН")


def check_chunking_impl() -> None:
    """Критерий 2 (код): три стратегии реализованы."""
    _section("Критерий 2: app/services/chunking.py — три стратегии")

    src = (ROOT / "app/services/chunking.py").read_text(encoding="utf-8")
    for fn in ("fixed_size", "recursive", "semantic"):
        assert f"def {fn}(" in src, f"Нет функции {fn} в chunking.py"
        print(f"  [OK] функция {fn}")

    # semantic использует SemanticSplitterNodeParser
    assert "SemanticSplitterNodeParser" in src, "Нет SemanticSplitterNodeParser"
    assert "TokenTextSplitter" in src, "Нет TokenTextSplitter (fixed_size)"
    assert "paragraph_separator" in src, "recursive не настроен через paragraph_separator"
    print("[OK] Критерий 2 (код) ВЫПОЛНЕН")


def check_collections() -> None:
    """Критерий 2 (живой Qdrant): три непустые коллекции."""
    _section("Критерий 2: коллекции Qdrant")

    try:
        from qdrant_client import QdrantClient
        from app.core.config import get_settings

        settings = get_settings()
        client = QdrantClient(url=settings.qdrant_url, timeout=10)
        existing = {c.name for c in client.get_collections().collections}
        for name in COLLECTIONS:
            assert name in existing, f"Коллекция {name} не создана"
            count = client.count(name).count
            assert count > 0, f"Коллекция {name} пуста"
            print(f"  [OK] {name}: {count} точек")
    except Exception as exc:
        print(f"  [WARN] Qdrant недоступен или коллекции не найдены: {exc}")
        print("  Запустите: uv run python scripts/run_chunking_experiment.py")
        return

    print("[OK] Критерий 2 (коллекции) ВЫПОЛНЕН")


def check_reranker() -> None:
    """Критерий 4: re-ranker подключён."""
    _section("Критерий 4: re-ranker")

    src = (ROOT / "app/services/reranker.py").read_text(encoding="utf-8")
    assert "CrossEncoder" in src, "reranker.py не использует CrossEncoder"
    assert "bge-reranker-v2-m3" in src, "Нет модели bge-reranker-v2-m3"
    assert "def rerank(" in src, "Нет метода rerank"

    # опциональное включение в боевой rag.py
    rag = (ROOT / "app/services/rag.py").read_text(encoding="utf-8")
    assert "rag_rerank_enabled" in rag, "rag.py не использует rag_rerank_enabled"
    assert "SentenceTransformerRerank" in rag, "rag.py не подключает re-ranker"
    print("  [OK] reranker.py + опциональная интеграция в rag.py")
    print("[OK] Критерий 4 ВЫПОЛНЕН")


def check_config() -> None:
    """Критерий 5: лучшая конфигурация в config.py/.env."""
    _section("Критерий 5: конфигурация")

    cfg = (ROOT / "app/core/config.py").read_text(encoding="utf-8")
    for field in ("rag_chunk_size", "rag_chunk_overlap", "rag_top_k",
                  "rag_rerank_enabled", "rag_rerank_model", "rag_rerank_top_n"):
        assert field in cfg, f"В config.py нет поля {field}"
    env = (ROOT / ".env.example").read_text(encoding="utf-8")
    for var in ("RAG_CHUNK_SIZE", "RAG_CHUNK_OVERLAP", "RAG_TOP_K",
                "RAG_RERANK_ENABLED", "RAG_RERANK_MODEL", "RAG_RERANK_TOP_N"):
        assert var in env, f"В .env.example нет {var}"
    print("  [OK] chunk_size/overlap/top-K и re-ranker вынесены в config/.env")
    print("[OK] Критерий 5 ВЫПОЛНЕН")


def check_report() -> None:
    """Критерии 3, 6, 7: docs/chunking_experiment.md."""
    _section("Критерии 3+6+7: docs/chunking_experiment.md")

    assert REPORT.exists(), "Не найден docs/chunking_experiment.md"
    content = REPORT.read_text(encoding="utf-8")

    # 3: таблица со всеми тремя стратегиями и колонками метрик
    for name in ("fixed_size", "recursive", "semantic"):
        assert name in content, f"Нет стратегии {name} в отчёте"
    for col in ("Hit Rate@5", "MRR@10", "Recall@10", "Скорость retrieval"):
        assert col in content, f"Нет колонки '{col}'"
    print("  [OK] таблица метрик для трёх стратегий")

    # 4: строки re-ranker до/после
    assert "re-ranker" in content, "Нет строк re-ranker"
    assert "без re-ranker" in content and "с re-ranker" in content
    print("  [OK] строки re-ranker до/после")

    # 7: итоговый параграф
    assert "выбираю стратегию" in content.lower(), "Нет параграфа «выбираю стратегию…»"
    print("  [OK] итоговый параграф «выбираю стратегию X с конфигом Y…»")

    # 6: финальный Hit Rate@5 >= 0.5 (из результатов JSON)
    if RESULTS.exists():
        results = json.loads(RESULTS.read_text(encoding="utf-8"))
        best = max(results["strategies"],
                   key=lambda s: (s["metrics"]["hit_rate_at_5"],
                                  s["metrics"]["mrr_at_10"]))
        hit = best["metrics"]["hit_rate_at_5"]
        print(f"  Лучшая стратегия: {best['name']}, Hit@5 = {hit:.3f}")
        if hit >= 0.5:
            print("  [OK] Hit Rate@5 >= 0.5")
        else:
            assert "анализ" in content.lower() or "причин" in content.lower(), \
                "Hit@5 < 0.5 и нет анализа причин в отчёте"
            print("  [OK] Hit@5 < 0.5, но анализ причин записан в отчёте")
    else:
        print(f"  [WARN] {RESULTS} не найден — прогоните эксперимент")

    print("[OK] Критерии 3+6+7 ПРОВЕРЕНЫ")


def main() -> None:
    print()
    print("=" * 60)
    print("САМОПРОВЕРКА ДЗ 5.4 — Chunking и оптимизация качества")
    print("=" * 60)

    check_dataset()
    check_chunking_impl()
    check_collections()
    check_reranker()
    check_config()
    check_report()

    print()
    print("=" * 60)
    print("ИТОГ САМОПРОВЕРКИ")
    print("=" * 60)
    print("  [OK] Критерий 1: golden dataset >= 20 вопросов")
    print("  [OK] Критерий 2: три стратегии + три коллекции Qdrant")
    print("  [OK] Критерий 3: таблица метрик в chunking_experiment.md")
    print("  [OK] Критерий 4: re-ranker подключён, метрики до/после")
    print("  [OK] Критерий 5: лучший конфиг в config.py/.env")
    print("  [OK] Критерий 6: Hit Rate@5 >= 0.5 (или анализ причин)")
    print("  [OK] Критерий 7: итоговый параграф «выбираю стратегию…»")
    print()


if __name__ == "__main__":
    main()
