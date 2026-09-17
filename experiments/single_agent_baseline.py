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

from app.core.config import get_settings
from experiments.kb_tool import search_knowledge_base
from experiments.measure_utils import run_one, save_results
from experiments.questions import QUESTIONS

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results.json"


async def main():
    settings = get_settings()
    # thinking отключён тем же флагом, что и в мультиагенте — иначе сравнение нечестное.
    model = ChatOpenAI(model="deepseek-v4-flash",
                           base_url=settings.llm.deepseek_base_url,
                           api_key=settings.llm.deepseek_api_key.get_secret_value(),
                           temperature=0,
                           extra_body={"thinking": {"type": "disabled"}})

    agent = create_agent(
        model=model, tools=[search_knowledge_base], name="assistant",
        system_prompt="Ты помощник по базе знаний. Найди факты через search_knowledge_base "
                      "и сразу собери связный ответ на русском с цитированием [1], [2]. "
                      "Если данных не нашлось — скажи «В базе знаний не нашлось ответа».",
    )

    records = [await run_one(agent, q, "single") for q in QUESTIONS]
    save_results(records, RESULTS)

if __name__ == "__main__":
    asyncio.run(main())