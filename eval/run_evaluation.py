#!/usr/bin/env python3
"""
Скрипт для запуска оценки модели на золотом наборе.
Использование:
    python eval/run_evaluation.py --golden eval/golden_dataset.json --judge gpt-4o-mini --out eval/runs/2026-06-19.json
"""

import asyncio
import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

import structlog
from openai import AsyncOpenAI

# Добавляем корневую папку проекта в sys.path для импорта модулей
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import Settings, LLMSettings
from app.services.llm import LLMService
from app.schemas.chat import ChatRequest, Message

# Настройка логирования (можно использовать существующую)
from app.observability.logger import setup_logging
setup_logging("INFO")
logger = structlog.get_logger()


def load_golden_dataset(path: Path) -> Dict[str, Any]:
    """Загружает golden dataset из JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


async def get_answer_from_service(service: LLMService, question: str) -> str:
    """Вызывает LLMService с вопросом и возвращает ответ."""
    req = ChatRequest(
        messages=[
            Message(role="system", content="Ты полезный ассистент, отвечающий на вопросы по документации."),
            Message(role="user", content=question),
        ],
        model=Settings().llm.default_model,  # используем модель из настроек
        temperature=0.0,  # детерминированный вывод
        stream=False,
    )
    response = await service.complete(req)
    return response.content


async def judge_answer(
    judge_client: AsyncOpenAI,
    judge_model: str,
    question: str,
    expected_answer: str,
    actual_answer: str,
    expected_keywords: List[str] = None,
    must_not_contain: List[str] = None,
) -> Dict[str, Any]:
    """
    Оценивает ответ с помощью judge-модели по критериям relevance, correctness, completeness.
    Возвращает словарь с полями: reasoning, relevance, correctness, completeness, explanation.
    """
    # Формируем промпт для G-Eval
    keywords_str = ", ".join(expected_keywords) if expected_keywords else "не указаны"
    must_not_str = ", ".join(must_not_contain) if must_not_contain else "не указаны"

    prompt = f"""Ты — эксперт по оценке качества ответов ИИ-ассистента на вопросы по технической документации.

Вопрос пользователя:
{question}

Ожидаемый (эталонный) ответ:
{expected_answer}

Фактический ответ модели:
{actual_answer}

Ожидаемые ключевые слова: {keywords_str}
Запрещённые фразы/данные (не должны присутствовать в ответе): {must_not_str}

Оцени ответ по трём критериям (каждый по шкале от 1 до 5, где 1 — плохо, 5 — отлично):
1. **Relevance** — насколько ответ соответствует вопросу и содержит ли он нужную информацию.
2. **Correctness** — насколько информация в ответе фактически верна, нет ошибок, совпадает ли с ожидаемым ответом.
3. **Completeness** — насколько ответ полный, охватывает ли все аспекты вопроса, содержит ли ключевые детали.

Сначала напиши краткое обоснование (reasoning) для каждой оценки, затем укажи оценки в формате JSON.
Требуемый формат ответа (JSON-объект):
{{
    "reasoning": "строка с обоснованием",
    "relevance": число от 1 до 5,
    "correctness": число от 1 до 5,
    "completeness": число от 1 до 5,
    "explanation": "краткое итоговое объяснение (одна строка)"
}}

Важно: Ответ должен быть только JSON-объектом, без лишнего текста."""
    try:
        response = await judge_client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        # Парсим JSON
        result = json.loads(content)
        # Приводим к числовым значениям
        for key in ["relevance", "correctness", "completeness"]:
            if key in result:
                result[key] = float(result[key])
            else:
                result[key] = 0.0
        return result
    except Exception as e:
        logger.error("Ошибка при вызове judge", error=str(e))
        return {
            "reasoning": f"Ошибка оценки: {e}",
            "relevance": 0.0,
            "correctness": 0.0,
            "completeness": 0.0,
            "explanation": "Ошибка при оценке"
        }


async def run_evaluation(
    golden_path: Path,
    judge_model: str,
    out_path: Path,
    settings: Settings,
):
    # Загружаем golden
    golden = load_golden_dataset(golden_path)
    version = golden.get("version", 1)
    items = golden.get("items", [])

    # Инициализируем клиент для тестируемой модели (используем тот же, что и в приложении)
    # Создаём HTTP-клиент (без прокси, если не нужен)
    import httpx
    http_client = httpx.AsyncClient(
        proxy=settings.proxy_url,
        timeout=httpx.Timeout(settings.llm.request_timeout, connect=5.0),
    )
    llm_client = AsyncOpenAI(
        api_key=settings.llm.openai_api_key.get_secret_value(),
        http_client=http_client,
        timeout=settings.llm.request_timeout,
        max_retries=settings.llm.max_retries,
    )
    # Кеш не используем в eval, чтобы всегда получать свежие ответы
    service = LLMService(llm=llm_client, cache=None, ttl=0)

    # Инициализируем judge-клиент (отдельный, можно использовать тот же клиент, но с другой моделью)
    judge_client = AsyncOpenAI(
        api_key=settings.llm.openai_api_key.get_secret_value(),
        http_client=http_client,
        timeout=settings.llm.request_timeout,
        max_retries=settings.llm.max_retries,
    )

    results = []
    for item in items:
        item_id = item.get("id", "unknown")
        question = item.get("question", "")
        expected_answer = item.get("expected_answer", "")
        expected_keywords = item.get("expected_keywords", [])
        must_not_contain = item.get("must_not_contain", [])

        logger.info(f"Обработка кейса {item_id}")
        try:
            actual_answer = await get_answer_from_service(service, question)
        except Exception as e:
            logger.error(f"Ошибка получения ответа для {item_id}", error=str(e))
            actual_answer = f"[Ошибка: {e}]"

        # Оценка судьёй
        scores = await judge_answer(
            judge_client,
            judge_model,
            question,
            expected_answer,
            actual_answer,
            expected_keywords,
            must_not_contain,
        )

        results.append({
            "id": item_id,
            "question": question,
            "answer": actual_answer,
            "scores": {
                "relevance": scores.get("relevance", 0.0),
                "correctness": scores.get("correctness", 0.0),
                "completeness": scores.get("completeness", 0.0),
            },
            "reasoning": scores.get("reasoning", ""),
            "explanation": scores.get("explanation", ""),
        })

    # Закрываем клиенты
    await http_client.aclose()
    await llm_client.close()
    await judge_client.close()

    # Вычисляем агрегаты
    if results:
        relevance_avg = sum(r["scores"]["relevance"] for r in results) / len(results)
        correctness_avg = sum(r["scores"]["correctness"] for r in results) / len(results)
        completeness_avg = sum(r["scores"]["completeness"] for r in results) / len(results)
        min_correctness = min(r["scores"]["correctness"] for r in results)
    else:
        relevance_avg = correctness_avg = completeness_avg = 0.0
        min_correctness = 0.0

    # Формируем итоговый словарь
    run_data = {
        "run_id": out_path.stem,  # например, "2026-06-19"
        "timestamp": datetime.now().isoformat(),
        "model_under_test": settings.llm.default_model,
        "judge_model": judge_model,
        "golden_version": version,
        "items": results,
        "aggregates": {
            "relevance_avg": relevance_avg,
            "correctness_avg": correctness_avg,
            "completeness_avg": completeness_avg,
            "min_correctness": min_correctness,
        }
    }

    # Сохраняем в JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(run_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Результаты сохранены в {out_path}")
    logger.info(f"Средние: relevance={relevance_avg:.2f}, correctness={correctness_avg:.2f}, completeness={completeness_avg:.2f}")
    logger.info(f"Минимальная correctness: {min_correctness:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Запуск оценки модели")
    parser.add_argument("--golden", required=True, help="Путь к golden_dataset.json")
    parser.add_argument("--judge", default="gpt-4o-mini", help="Модель для оценки (judge)")
    parser.add_argument("--out", required=True, help="Путь для сохранения результатов (JSON)")
    args = parser.parse_args()

    # Загружаем настройки
    settings = Settings()
    # Убедимся, что модель тестирования берётся из настроек (можно переопределить через env)

    # Создаём папку для out, если её нет
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    asyncio.run(run_evaluation(
        golden_path=Path(args.golden),
        judge_model=args.judge,
        out_path=out_path,
        settings=settings,
    ))


if __name__ == "__main__":
    main()