import asyncio
from pathlib import Path

# TLS-инспекция Kaspersky подменяет сертификаты: certifi (httpx/openai) их не признаёт,
# а хранилище сертификатов Windows — признаёт. inject_into_ssl() переключает Python на него.
import truststore

truststore.inject_into_ssl()

# Тихий HuggingFace: langchain тянет transformers/huggingface_hub за собой,
# поэтому переменные надо выставить до его импорта (см. app/core/hf_env.py).
import app.core.hf_env  # noqa: E402, F401

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_supervisor import create_supervisor

from app.core.config import get_settings
from experiments.kb_tool import search_knowledge_base
from experiments.measure_utils import run_one, save_results
from experiments.questions import QUESTIONS

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results.json"
ARCH = ROOT / "docs" / "architecture-multi-agent.md"

MERMAID_BEGIN = "<!-- BEGIN: draw_mermaid -->"
MERMAID_END = "<!-- END: draw_mermaid -->"


def save_architecture(path: Path, mermaid: str) -> None:
    """Вставляет схему в документ по маркерам, не затирая пояснительный текст вокруг."""
    block = f"{MERMAID_BEGIN}\n```mermaid\n{mermaid}\n```\n{MERMAID_END}"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if MERMAID_BEGIN in text and MERMAID_END in text:
            start = text.index(MERMAID_BEGIN)
            end = text.index(MERMAID_END) + len(MERMAID_END)
            path.write_text(text[:start] + block + text[end:], encoding="utf-8")
            return
    path.write_text(block + "\n", encoding="utf-8")


async def main() -> None:
    settings = get_settings()
    # thinking отключён: supervisor пересобирает историю сообщений и теряет reasoning_content,
    # из-за чего DeepSeek отвечает 400 «reasoning_content ... must be passed back to the API».
    model = ChatOpenAI(model="deepseek-v4-flash",
                       base_url=settings.llm.deepseek_base_url,
                       api_key=settings.llm.deepseek_api_key.get_secret_value(),
                       temperature=0,
                       extra_body={"thinking": {"type": "disabled"}})

    researcher = create_agent(
        model=model, tools=[search_knowledge_base], name="researcher",
        system_prompt=(
            "Ты исследователь. Собери факты через search_knowledge_base и верни "
            "маркированный список фактов с указанием источника и score. "
            "Нумеруй источники по порядку: [1] — имя файла, [2] — имя файла; "
            "нумерацию сохраняй до конца работы и включай только те источники, "
            "из которых реально взяты факты. "
            "Финальный ответ пользователю НЕ пиши — это делает writer."
        ),
    )

    writer = create_agent(
        model=model, tools=[], name="writer",
        system_prompt=(
            "Ты редактор. По списку фактов от researcher собери связный ответ на русском. "
            "Каждый факт сопровождай ссылкой [1], [2] по номеру источника. "
            "Не добавляй вступлений и заключений вида «задача выполнена», «итог», «оба этапа завершены» — "
            "только ответ по существу. "
            "Если фактов нет или researcher сообщил, что данных не нашлось — напиши "
            "«В базе знаний не нашлось ответа»."
        ),
    )

    app = create_supervisor(
        agents=[researcher, writer], model=model,
        prompt=(
            "Ты супервизор команды из researcher и writer. Жёсткие правила:\n"
            "1. Ты НИКОГДА не пишешь ответ по существу вопроса сам — его готовит writer.\n"
            "2. Первый шаг — всегда передать задачу researcher для сбора фактов.\n"
            "3. Получив факты от researcher, ты ОБЯЗАН передать работу writer и завершить свой ход.\n"
            "4. Когда writer вернул финальный ответ, верни его пользователю ДОСЛОВНО, слово в слово, "
            "вместе со ссылками [1], [2]. Не добавляй комментариев, вступлений и заключений "
            "вида «работа завершена», «задача передана writer'у»."
        ),
        output_mode="last_message",
    ).compile(checkpointer=InMemorySaver())

    records = [await run_one(app, q, "multi") for q in QUESTIONS]
    save_results(records, RESULTS)

    save_architecture(ARCH, app.get_graph().draw_mermaid())


if __name__ == "__main__":
    asyncio.run(main())