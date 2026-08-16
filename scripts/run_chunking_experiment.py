"""Эксперимент по чанкингу (ДЗ 5.4).

Один корпус (data/rag-block-03) → три стратегии чанкинга → три коллекции
Qdrant (docs_fixed / docs_recursive / docs_semantic) → retrieval-метрики
(Hit Rate@5, MRR@10, Recall@10) на golden dataset (tests/eval/retrieval_dataset.json)
→ re-ranker (bge-reranker-v2-m3) → подбор (chunk_size, overlap, top-K).

Требует живой Qdrant (QDRANT_URL). LLM не нужен. Результаты пишутся в
docs/chunking_experiment.md и var/chunking_results.json.

Запуск:
    uv run python scripts/run_chunking_experiment.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.core.config import get_settings
from app.services import chunking
from app.services.retrieval_eval import evaluate_retrieval
from app.services.reranker import Reranker

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

GOLDEN_PATH = Path("tests/eval/retrieval_dataset.json")
RESULTS_JSON = Path("var/chunking_results.json")
REPORT_MD = Path("docs/chunking_experiment.md")

# Основное сравнение: фиксированные параметры для каждой стратегии (шаг 2 задания).
STRATEGIES = [
    (
        "fixed_size",
        "docs_fixed",
        lambda docs, em: chunking.fixed_size(docs, chunk_size=512, chunk_overlap=64),
        {"chunk_size": 512, "chunk_overlap": 64},
    ),
    (
        "recursive",
        "docs_recursive",
        lambda docs, em: chunking.recursive(docs, chunk_size=512, chunk_overlap=64),
        {"chunk_size": 512, "chunk_overlap": 64},
    ),
    (
        "semantic",
        "docs_semantic",
        lambda docs, em: chunking.semantic(
            docs, em, buffer_size=1, breakpoint_percentile_threshold=95
        ),
        {"buffer_size": 1, "breakpoint_percentile_threshold": 95},
    ),
]

# Тюнинг лучшей стратегии: 2×2 сетка chunk_size × overlap (шаг 7 задания).
TUNING_GRID = [
    {"chunk_size": 256, "chunk_overlap": 32},
    {"chunk_size": 256, "chunk_overlap": 64},
    {"chunk_size": 512, "chunk_overlap": 32},
    {"chunk_size": 512, "chunk_overlap": 64},
]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _doc_id(node: NodeWithScore) -> str:
    """Достаёт имя файла (doc_id) из метаданных ноды."""
    md = (node.node.metadata or {}) if hasattr(node, "node") else (node.metadata or {})
    return md.get("file_name") or str(md.get("file_path", "")).rsplit("/", 1)[-1]


def _node_text(node: NodeWithScore) -> str:
    return node.node.get_content() if hasattr(node, "node") else node.get_content()


def _load_golden() -> list[dict]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _recreate_collection(client: QdrantClient, name: str) -> None:
    if client.collection_exists(name):
        client.delete_collection(name)


def _index_nodes(
    client: QdrantClient,
    nodes: list,
    collection: str,
) -> VectorStoreIndex:
    """Индексирует готовые ноды в коллекцию Qdrant (через Settings.embed_model)."""
    vector_store = QdrantVectorStore(collection_name=collection, client=client)
    storage = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex(nodes, storage_context=storage)


def _retrieve_candidates(retriever, query: str, top_k: int) -> list[dict]:
    nodes = retriever.retrieve(query)
    out = []
    for n in nodes[:top_k]:
        out.append(
            {
                "doc_id": _doc_id(n),
                "text": _node_text(n),
                "score": float(n.score or 0.0),
            }
        )
    return out


def _retrieve_doc_ids(retriever, query: str) -> list[str]:
    return [c["doc_id"] for c in _retrieve_candidates(retriever, query, top_k=10)]


def _measure_latency_ms(retriever, golden: list[dict], warmup: int = 3) -> float:
    """Среднее время retrieval (query → кандидаты) по всем вопросам, мс."""
    # прогрев (загрузка модели/кэшей)
    for q in golden[:warmup]:
        retriever.retrieve(q["question"])
    times = []
    for item in golden:
        t0 = time.perf_counter()
        retriever.retrieve(item["question"])
        times.append((time.perf_counter() - t0) * 1000.0)
    return round(sum(times) / len(times), 1)


def _run_strategy(
    client: QdrantClient,
    embed_model,
    name: str,
    collection: str,
    chunk_fn,
    chunk_kwargs: dict,
    documents: list,
    n_files: int,
    golden: list[dict],
) -> dict:
    _log(f"\n[{name}] чанкинг...")
    nodes = chunk_fn(documents, embed_model)
    stats = chunking.chunk_stats(nodes, n_documents=n_files)
    _log(f"[{name}] чанков: {stats['total_chunks']}, "
         f"средн. длина {stats['avg_chunk_len_chars']} симв., "
         f"средн. {stats['avg_chunks_per_doc']} чанка/файл")

    _log(f"[{name}] индексация в {collection}...")
    _recreate_collection(client, collection)
    index = _index_nodes(client, nodes, collection)
    retriever = index.as_retriever(similarity_top_k=10)

    _log(f"[{name}] оценка на golden dataset...")
    metrics = evaluate_retrieval(
        lambda q: _retrieve_doc_ids(retriever, q), golden, top_k=10
    )
    latency_ms = _measure_latency_ms(retriever, golden)

    _log(f"[{name}] Hit@5={metrics['hit_rate_at_5']:.3f} "
         f"MRR@10={metrics['mrr_at_10']:.3f} "
         f"Recall@10={metrics['recall_at_10']:.3f} "
         f"latency={latency_ms} мс")
    return {
        "name": name,
        "collection": collection,
        "chunk_kwargs": chunk_kwargs,
        "stats": stats,
        "metrics": metrics,
        "latency_ms": latency_ms,
    }


def _run_tuning(
    client: QdrantClient,
    embed_model,
    best_name: str,
    best_chunk_fn,
    documents: list,
    n_files: int,
    golden: list[dict],
) -> list[dict]:
    """4 конфига чанкинга × 2 top-K → строки тюнинга."""
    rows = []
    for i, cfg in enumerate(TUNING_GRID):
        collection = f"docs_tune_{i}"
        _log(f"\n[tuning {i}] chunk_size={cfg['chunk_size']} "
             f"overlap={cfg['chunk_overlap']}...")
        nodes = best_chunk_fn(documents, chunk_size=cfg["chunk_size"],
                              chunk_overlap=cfg["chunk_overlap"])
        stats = chunking.chunk_stats(nodes, n_documents=n_files)
        _recreate_collection(client, collection)
        index = _index_nodes(client, nodes, collection)

        for top_k in (10, 20):
            retriever = index.as_retriever(similarity_top_k=top_k)
            metrics = evaluate_retrieval(
                lambda q: _retrieve_doc_ids(retriever, q), golden, top_k=top_k
            )
            latency_ms = _measure_latency_ms(retriever, golden)
            rows.append(
                {
                    "strategy": best_name,
                    "chunk_size": cfg["chunk_size"],
                    "chunk_overlap": cfg["chunk_overlap"],
                    "top_k": top_k,
                    "avg_chunk_len_chars": stats["avg_chunk_len_chars"],
                    "hit_rate_at_5": metrics["hit_rate_at_5"],
                    "mrr_at_10": metrics["mrr_at_10"],
                    "recall_at_10": metrics["recall_at_10"],
                    "latency_ms": latency_ms,
                }
            )
            _log(f"  top_k={top_k}: Hit@5={metrics['hit_rate_at_5']:.3f} "
                 f"MRR@10={metrics['mrr_at_10']:.3f} "
                 f"Recall@10={metrics['recall_at_10']:.3f} "
                 f"latency={latency_ms} мс")
    return rows


def _run_reranker(
    client: QdrantClient,
    embed_model,
    best_name: str,
    best_chunk_fn,
    documents: list,
    n_files: int,
    golden: list[dict],
    rerank_model: str,
) -> dict:
    """Лучшая стратегия до/после re-ranker (пересортировка тех же top-20)."""
    collection = "docs_rerank_best"
    _log(f"\n[reranker] индексация лучшей стратегии {best_name}...")
    nodes = best_chunk_fn(documents, chunk_size=512, chunk_overlap=64)
    _recreate_collection(client, collection)
    index = _index_nodes(client, nodes, collection)
    retriever = index.as_retriever(similarity_top_k=20)

    _log("[reranker] загрузка cross-encoder...")
    reranker = Reranker(model_name=rerank_model, top_n=None)

    before_hit = before_mrr = before_recall = 0.0
    after_hit = after_mrr = after_recall = 0.0
    for item in golden:
        question = item["question"]
        relevant = item["relevant_doc_ids"]
        candidates = _retrieve_candidates(retriever, question, top_k=20)
        before_ids = [c["doc_id"] for c in candidates]

        from app.services.retrieval_eval import hit_rate_at_k, mrr_at_k, recall_at_k

        before_hit += hit_rate_at_k(before_ids, relevant, 5)
        before_mrr += mrr_at_k(before_ids, relevant, 10)
        before_recall += recall_at_k(before_ids, relevant, 10)

        ranked = reranker.rerank(question, candidates)
        after_ids = [c["doc_id"] for c in ranked]
        after_hit += hit_rate_at_k(after_ids, relevant, 5)
        after_mrr += mrr_at_k(after_ids, relevant, 10)
        after_recall += recall_at_k(after_ids, relevant, 10)

    n = len(golden) or 1
    return {
        "strategy": best_name,
        "model": rerank_model,
        "before": {
            "hit_rate_at_5": round(before_hit / n, 4),
            "mrr_at_10": round(before_mrr / n, 4),
            "recall_at_10": round(before_recall / n, 4),
        },
        "after": {
            "hit_rate_at_5": round(after_hit / n, 4),
            "mrr_at_10": round(after_mrr / n, 4),
            "recall_at_10": round(after_recall / n, 4),
        },
    }


def _write_report(results: dict) -> None:
    lines: list[str] = []
    lines.append("# Chunking-эксперимент — ДЗ 5.4\n")
    lines.append("> Автогенерировано `scripts/run_chunking_experiment.py`.\n")

    corpus = results["corpus"]
    golden = results["golden"]
    lines.append("## Условия эксперимента\n")
    lines.append(f"- Корпус: `data/rag-block-03` — {corpus['n_files']} файлов "
                 f"({corpus['n_llamaindex_docs']} llama-index-документов после постраничного чтения PDF).\n")
    lines.append(f"- Эмбеддинги: `{results['embed_model']}`.\n")
    lines.append(f"- Golden dataset: {golden['n_questions']} вопросов "
                 f"(`tests/eval/retrieval_dataset.json`).\n")
    lines.append(f"- Re-ranker: `{results['rerank_model']}` (cross-encoder, локально).\n")

    lines.append("\n## Чанкинг: статистика\n")
    lines.append("| Стратегия | Всего чанков | Чанков/файл | Средняя длина (симв.) |\n")
    lines.append("|---|---|---|---|\n")
    for s in results["strategies"]:
        st = s["stats"]
        lines.append(f"| {s['name']} | {st['total_chunks']} | "
                     f"{st['avg_chunks_per_doc']} | {st['avg_chunk_len_chars']} |\n")

    lines.append("\n## Retrieval-метрики (top-K=10)\n")
    lines.append("| Стратегия | Hit Rate@5 | MRR@10 | Recall@10 | Средняя длина chunk | Скорость retrieval (мс) |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for s in results["strategies"]:
        m = s["metrics"]
        lines.append(f"| {s['name']} | {m['hit_rate_at_5']:.3f} | {m['mrr_at_10']:.3f} | "
                     f"{m['recall_at_10']:.3f} | {s['stats']['avg_chunk_len_chars']} | "
                     f"{s['latency_ms']} |\n")

    lines.append("\n## Re-ranker (до/после, лучшая стратегия)\n")
    r = results["reranker"]
    lines.append(f"Стратегия: `{r['strategy']}`, модель: `{r['model']}`.\n")
    lines.append("| Вариант | Hit Rate@5 | MRR@10 | Recall@10 |\n")
    lines.append("|---|---|---|---|\n")
    lines.append(f"| без re-ranker | {r['before']['hit_rate_at_5']:.3f} | "
                 f"{r['before']['mrr_at_10']:.3f} | {r['before']['recall_at_10']:.3f} |\n")
    lines.append(f"| с re-ranker | {r['after']['hit_rate_at_5']:.3f} | "
                 f"{r['after']['mrr_at_10']:.3f} | {r['after']['recall_at_10']:.3f} |\n")

    lines.append("\n## Тюнинг (chunk_size × overlap × top-K)\n")
    lines.append("| chunk_size | overlap | top-K | Hit Rate@5 | MRR@10 | Recall@10 | Средняя длина | Скорость (мс) |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")
    for t in results["tuning"]:
        lines.append(f"| {t['chunk_size']} | {t['chunk_overlap']} | {t['top_k']} | "
                     f"{t['hit_rate_at_5']:.3f} | {t['mrr_at_10']:.3f} | "
                     f"{t['recall_at_10']:.3f} | {t['avg_chunk_len_chars']} | "
                     f"{t['latency_ms']} |\n")

    lines.append("\n## Вывод\n")
    lines.append("<!-- TODO: итоговый параграф «выбираю стратегию X с конфигом Y, потому что…» "
                "заполняется после анализа чисел. -->\n")

    REPORT_MD.write_text("".join(lines), encoding="utf-8")
    _log(f"\nОтчёт записан: {REPORT_MD}")


def main() -> None:
    settings = get_settings()
    _log("=" * 60)
    _log("Chunking-эксперимент ДЗ 5.4")
    _log("=" * 60)

    golden = _load_golden()
    _log(f"Golden dataset: {len(golden)} вопросов")

    embed_model = chunking.build_embed_model(settings.embedding.model)
    Settings.embed_model = embed_model

    documents = chunking.load_documents(str(settings.rag_data_dir), recursive=True)
    n_files = len({d.metadata.get("file_name") for d in documents})
    _log(f"Корпус: {n_files} файлов, {len(documents)} llama-index-документов")

    client = QdrantClient(url=settings.qdrant_url)

    strategies = []
    for name, collection, chunk_fn, kwargs in STRATEGIES:
        strategies.append(
            _run_strategy(client, embed_model, name, collection, chunk_fn, kwargs,
                          documents, n_files, golden)
        )

    # Лучшая стратегия по Hit Rate@5 (tie-break — по MRR@10).
    best = max(strategies, key=lambda s: (s["metrics"]["hit_rate_at_5"],
                                          s["metrics"]["mrr_at_10"]))
    best_name = best["name"]
    _log(f"\nЛучшая стратегия: {best_name}")

    # chunk-функция лучшей стратегии для тюнинга и re-ranker
    if best_name == "recursive":
        best_chunk_fn = lambda docs, **kw: chunking.recursive(docs, **kw)
    elif best_name == "fixed_size":
        best_chunk_fn = lambda docs, **kw: chunking.fixed_size(docs, **kw)
    else:
        best_chunk_fn = lambda docs, **kw: chunking.semantic(
            docs, embed_model, buffer_size=1, breakpoint_percentile_threshold=95
        )

    tuning = _run_tuning(client, embed_model, best_name, best_chunk_fn,
                         documents, n_files, golden)
    reranker = _run_reranker(client, embed_model, best_name, best_chunk_fn,
                             documents, n_files, golden, settings.rag_rerank_model)

    results = {
        "corpus": {"n_files": n_files, "n_llamaindex_docs": len(documents)},
        "embed_model": settings.embedding.model,
        "rerank_model": settings.rag_rerank_model,
        "golden": {"n_questions": len(golden)},
        "strategies": strategies,
        "reranker": reranker,
        "tuning": tuning,
    }

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    _log(f"Результаты записаны: {RESULTS_JSON}")
    _write_report(results)

    _log("\nГотово.")


if __name__ == "__main__":
    main()
