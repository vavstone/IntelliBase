"""LLM-судья для ДЗ 6.5: оценка качества ответов single-agent vs multi-agent.

Что делает:
1. читает `experiments/results.json` (10 записей: 5 вопросов x 2 реализации);
2. для каждого ответа заново достаёт контекст тем же ретривером, что и агенты
   (`RAGService.retrieve`), с полными текстами чанков — сниппетов по 200 символов
   судье не хватает, и он помечает реальные факты как выдуманные;
3. просит DeepSeek оценить ответ по трём критериям (1–5) ВСЛЕПУЮ: в промпте нет
   пометки, какая реализация дала ответ;
4. дописывает в `results.json` поля `quality_score`, `grounded`, `citations`, `honesty`, `comment`.

Рубрика (по одному баллу на критерий):
- grounded  — опирается ли ответ на контекст, нет ли фактов «из головы»;
- citations — есть ли ссылки [1], [2] и указывают ли они на реальные фрагменты;
- honesty   — если ответа в контексте нет, честно ли система об этом сказала.

Итоговый `quality_score` — среднее трёх критериев (1–5).

Запуск:
    uv run python -m experiments.judge
"""

import asyncio
import json
import random
import re
import sys
from pathlib import Path

# TLS-инспекция Kaspersky подменяет сертификаты: certifi (httpx/openai) их не признаёт,
# а хранилище сертификатов Windows — признаёт. inject_into_ssl() переключает Python на него.
import truststore

truststore.inject_into_ssl()

# Тихий HuggingFace: langchain тянет transformers/huggingface_hub за собой,
# поэтому переменные надо выставить до его импорта (см. app/core/hf_env.py).
import app.core.hf_env  # noqa: E402, F401

from langchain_openai import ChatOpenAI  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.services.rag import RAGService  # noqa: E402
from experiments.questions import QUESTIONS  # noqa: E402

_RAG: RAGService | None = None

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results.json"

CRITERIA = ("grounded", "citations", "honesty")

SYSTEM_PROMPT = (
    "Ты — судья качества ответов RAG-системы по базе знаний ФТС. "
    "Оцениваешь ТОЛЬКО присланный ответ, не додумывая ничего от себя.\n\n"
    "Критерии (каждый — целое число от 1 до 5):\n"
    "1. grounded — опирается ли ответ на предоставленный контекст; нет ли фактов, "
    "которых в контексте нет (5 — всё из контекста, 1 — выдумано).\n"
    "2. citations — есть ли ссылки вида [1], [2] на фрагменты контекста и указывают ли они "
    "на реально использованные фрагменты (5 — есть и корректны, 1 — нет ссылок).\n"
    "3. honesty — если ответа в контексте нет, честно ли система об этом сказала. "
    "Если ответ в контексте есть — оцени, не добавлены ли оговорки-галлюцинации "
    "(5 — честно и по делу, 1 — выдумала ответ или отговорилась при наличии данных).\n\n"
    "Верни СТРОГО JSON без пояснений вокруг: "
    '{"grounded": N, "citations": N, "honesty": N, "comment": "одна короткая фраза"}'
)


def _parse_scores(text: str) -> dict:
    """Достаёт JSON из ответа судьи (на случай ```-обёртки)."""
    match = re.search(r"\{.*\}", text, re.S)
    assert match, f"судья вернул не JSON: {text[:200]}"
    data = json.loads(match.group(0))
    for key in CRITERIA:
        value = int(data[key])
        assert 1 <= value <= 5, f"{key}={value} вне диапазона 1..5"
        data[key] = value
    return data


async def _context_for(question: str) -> str:
    """Контекст для судьи: тот же ретривер, что и у агентов, но с полными текстами чанков.

    Через `search_knowledge_base` брать нельзя: он отдаёт `snippet` по 200 символов,
    и судья помечает реальные факты как выдуманные.
    """
    global _RAG
    if _RAG is None:
        settings = get_settings()
        _RAG = RAGService(settings)
        _RAG.build()
    nodes = await _RAG.retrieve(question)
    parts: list[str] = []
    for i, node in enumerate(nodes, start=1):
        meta = node.metadata or {}
        name = meta.get("source") or meta.get("file_name") or "unknown"
        parts.append(f"[{i}] {name} (score={round(node.score or 0.0, 3)})\n{node.text[:1500].strip()}")
    if not parts:
        return "(контекст пуст: релевантных фрагментов не найдено)"
    return "\n\n".join(parts)


async def judge_one(model: ChatOpenAI, question: str, context: str, answer: str) -> dict:
    user = (
        f"Вопрос пользователя:\n{question}\n\n"
        f"Контекст, выданный системе:\n{context}\n\n"
        f"Оцениваемый ответ:\n{answer}"
    )
    response = await model.ainvoke([("system", SYSTEM_PROMPT), ("user", user)])
    scores = _parse_scores(str(response.content))
    scores["quality_score"] = round(sum(scores[k] for k in CRITERIA) / len(CRITERIA), 2)
    return scores


async def main() -> None:
    settings = get_settings()
    # thinking отключён: у судьи не должно быть «рассуждений» в ответе, нужен строгий JSON
    model = ChatOpenAI(model="deepseek-v4-flash",
                       base_url=settings.llm.deepseek_base_url,
                       api_key=settings.llm.deepseek_api_key.get_secret_value(),
                       temperature=0,
                       extra_body={"thinking": {"type": "disabled"}})

    records = json.loads(RESULTS.read_text(encoding="utf-8"))
    print(f"записей на оценку: {len(records)}\n")

    question_by_id = {q["id"]: q["text"] for q in QUESTIONS}

    # порядок оценки перемешан, чтобы реализация не читалась по последовательности
    order = list(range(len(records)))
    random.shuffle(order)

    contexts: dict[str, str] = {}
    for idx in order:
        record = records[idx]
        question = question_by_id[record["qid"]]
        if question not in contexts:
            contexts[question] = await _context_for(question)

        scores = await judge_one(model, question, contexts[question], str(record.get("answer", "")))
        record.update(scores)
        label = f"{record['impl']}/{record['qid']}"
        print(f"  {label:<14} grounded={scores['grounded']} citations={scores['citations']} "
              f"honesty={scores['honesty']} -> {scores['quality_score']}  ({scores['comment'][:60]})")

    RESULTS.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nИТОГ (среднее по 5 вопросам)")
    print(f"  {'метрика':<12}{'single':>10}{'multi':>10}")
    for key in CRITERIA + ("quality_score",):
        row = {}
        for impl in ("single", "multi"):
            values = [r[key] for r in records if r["impl"] == impl]
            row[impl] = sum(values) / len(values) if values else 0.0
        print(f"  {key:<12}{row['single']:>10.2f}{row['multi']:>10.2f}")
    print(f"\nоценки записаны в {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
