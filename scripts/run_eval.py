"""Прогон RAGAS-метрик по golden dataset через текущий RAG (группа eval).

Считает пять метрик на строку: faithfulness, answer_relevancy, context_precision,
context_recall (collections) и has_citation (@discrete_metric), плюс latency_ms.
По датасету идём конкурентно через asyncio.gather (с семафором, чтобы не
зафлудить локальную Ollama), агрегаты и per-row собираем в pandas.DataFrame и
пишем в tests/eval/results/{timestamp}_{label}.csv — это audit log, по которому
строится временной ряд метрик.

A/B-варианты задаются аргументами --collection / --top-k / --rerank (override
конфига), чтобы один и тот же скрипт воспроизводил baseline и эксперименты без
жонглирования переменными окружения. Судья (DeepSeek по умолчанию) отделён от
production-LLM в /rag/query.

Запуск:
    uv run --extra eval python scripts/run_eval.py --label baseline
    uv run --extra eval python scripts/run_eval.py --collection rag_block_05_chunk1024 --label chunk_1024
    uv run --extra eval python scripts/run_eval.py --top-k 5 --label top_k_5
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.eval.metrics import (  # noqa: E402
    build_embeddings,
    build_judge,
    build_metrics,
    eval_row,
    make_has_citation,
)
from app.services.rag import RAGService  # noqa: E402


async def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS eval по golden dataset")
    parser.add_argument("--golden", default="tests/eval/golden_dataset.json")
    parser.add_argument(
        "--label", default="baseline", help="метка конфигурации: baseline, chunk_1024, top_k_5 ..."
    )
    parser.add_argument("--out-dir", default="tests/eval/results")
    parser.add_argument("--collection", default=None, help="override RAG_COLLECTION")
    parser.add_argument("--top-k", type=int, default=None, help="override RAG_TOP_K")
    parser.add_argument("--rerank", action="store_true", help="включить реранкер")
    parser.add_argument(
        "--concurrency", type=int, default=2, help="параллельных RAG-вызовов (Ollama на CPU)"
    )
    parser.add_argument(
        "--per-question-timeout", type=float, default=600.0,
        help="таймаут на один вопрос (сек) — защита от зависшего вызова судьи",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.collection:
        settings = settings.model_copy(update={"rag_collection": args.collection})
    if args.top_k is not None:
        settings = settings.model_copy(update={"rag_top_k": args.top_k})
    if args.rerank:
        settings = settings.model_copy(update={"rag_rerank_enabled": True})

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    print(f"Загружено {len(golden)} пар из {args.golden}")
    print(
        f"Конфигурация: collection={settings.rag_collection} top_k={settings.rag_top_k} "
        f"rerank={settings.rag_rerank_enabled} judge={settings.eval_judge_model}"
    )

    rag = RAGService(settings)
    await asyncio.to_thread(rag.build)

    judge = build_judge(settings)
    embeddings = build_embeddings(settings)
    metrics = build_metrics(judge, embeddings)
    has_citation = make_has_citation(judge)

    # Семафор ограничивает число одновременных RAG-генераций: локальная Ollama
    # на CPU обрабатывает запросы последовательно, и без лимита хвост очереди
    # вылез бы за RAG_LLM_TIMEOUT.
    sem = asyncio.Semaphore(args.concurrency)
    done_count = 0

    async def _guarded(row: dict) -> dict:
        nonlocal done_count
        async with sem:
            try:
                result = await asyncio.wait_for(
                    eval_row(rag, row, metrics, has_citation),
                    timeout=args.per_question_timeout,
                )
            except asyncio.TimeoutError:
                result = {"user_input": row.get("user_input", ""), "error": "timeout"}
            done_count += 1
            print(f"[{done_count}/{len(golden)}] {row.get('user_input', '')[:60]}", flush=True)
            return result

    try:
        rows = await asyncio.gather(
            *[_guarded(row) for row in golden], return_exceptions=True
        )
    finally:
        await rag.close()

    # Упавшая строка не должна ронять весь прогон: помечаем её пустой строкой.
    safe_rows: list[dict] = []
    for i, r in enumerate(rows):
        if isinstance(r, BaseException):
            safe_rows.append({"user_input": golden[i].get("user_input", ""), "error": str(r)})
        else:
            safe_rows.append(r)

    df = pd.DataFrame(safe_rows)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out = Path(args.out_dir) / f"{stamp}_{args.label}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    counter = getattr(judge, "token_counter", None)

    # Sidecar с конфигом прогона — часть audit-log: по нему восстанавливается,
    # на каком chunk_size/top_k/re-ranker/судье получены эти числа.
    meta = {
        "label": args.label,
        "timestamp": stamp,
        "collection": settings.rag_collection,
        "chunk_size": settings.rag_chunk_size,
        "chunk_overlap": settings.rag_chunk_overlap,
        "top_k": settings.rag_top_k,
        "rerank_enabled": settings.rag_rerank_enabled,
        "rag_llm_model": settings.rag_llm_model,
        "embedding_model": settings.embedding.model,
        "judge_provider": settings.eval_judge_provider,
        "judge_model": settings.eval_judge_model,
        "n_rows": len(df),
        "judge_prompt_tokens": counter.prompt_tokens if counter else None,
        "judge_completion_tokens": counter.completion_tokens if counter else None,
    }
    meta_path = Path(args.out_dir) / f"{stamp}_{args.label}.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if counter is not None:
        print(
            f"\nТокены судьи ({settings.eval_judge_model}): "
            f"input={counter.prompt_tokens} output={counter.completion_tokens} "
            f"итого={counter.total}"
        )

    print(f"\nРезультат: {out}")
    numeric = df.select_dtypes(include="number")
    print("\nАгрегаты (средние):")
    print(numeric.mean(numeric_only=True))
    if "has_citation" in df.columns:
        yes = (df["has_citation"] == "yes").mean()
        print(f"\nhas_citation (доля yes): {yes:.3f}")
    print("\nТоп худших по faithfulness:")
    if "faithfulness" in df.columns:
        worst = df.sort_values("faithfulness").head()
        print(worst[["user_input", "faithfulness"]].to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())
