"""Демо-прогон RAG с Phoenix-трейсингом LlamaIndex (20+ запросов).

Регистрирует LlamaIndexInstrumentor и прогоняет разнообразные вопросы через
RAGService, чтобы в Phoenix (http://localhost:6006) появились древовидные трейсы:
retriever-спаны с similarity scores и LLM-спаны с prompt/usage. Часть вопросов —
из golden dataset, часть — свободные.

Запуск (нужен поднятый phoenix из compose.yaml):
    PHOENIX_ENABLED=true uv run --extra tracing python scripts/trace_demo.py
"""

import asyncio
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.observability.tracing import setup_tracing  # noqa: E402
from app.services.rag import RAGService  # noqa: E402

# Свободные вопросы по домену ФТС + пара вне-базы (для спанов отказа).
FREE_QUESTIONS = [
    "Что входит в состав КПС «Тарифы — Реестр ОИС»?",
    "Какие методики планового мониторинга используются в ИСС «Малахит»?",
    "Как выполняется верификация сертификатов о происхождении товаров?",
    "Что такое параллельный импорт и как формируется витрина?",
    "Какие изменения вносятся в форму 38-запрет?",
    "Как оформляется приостановление выпуска товаров с объектами интеллектуальной собственности?",
    "Какие документы нужны для гарантийного ремонта?",
    "Что регулирует приказ ФТС № 665?",
    "Как подать заявку на включение объекта в таможенный реестр?",
    "Какие показатели планового мониторинга входят в методику 14?",
    "Что такое ТРОИС и как в нём проходит экспертиза?",
    "Как формируются сертификаты согласования?",
    "Какая завтра погода в Москве?",
    "Как приготовить плов?",
]


def _golden_questions(n: int = 8) -> list[str]:
    golden_path = Path("tests/eval/golden_dataset.json")
    if not golden_path.exists():
        return []
    import json

    data = json.loads(golden_path.read_text(encoding="utf-8"))
    return [item["user_input"] for item in data[:n]]


async def main() -> None:
    settings = get_settings()
    if not setup_tracing(settings):
        print("Трейсинг не включён — задай PHOENIX_ENABLED=true")
        return

    rag = RAGService(settings)
    await asyncio.to_thread(rag.build)

    questions = _golden_questions() + FREE_QUESTIONS
    print(f"Прогоняю {len(questions)} запросов через RAG с трейсингом...\n")
    for q in questions:
        result = await rag.evaluate_inputs(q)
        print(f">> {q}\n   confident={result['confident']} "
              f"top_score={result['top_score']} answer={result['answer'][:70]!r}")

    await rag.close()
    print("\nЖду flush BatchSpanProcessor (5 c)...")
    time.sleep(5)
    print("Готово. Открой http://localhost:6006 → Traces.")


if __name__ == "__main__":
    asyncio.run(main())
