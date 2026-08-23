import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable
from openai import OpenAI,APITimeoutError

from app.core.config import get_settings
from app.tools.react_tools import TOOLS, DISPATCH

logger = logging.getLogger(__name__)


def _accumulate_usage(total: dict, usage) -> None:
    if usage is None:
        return
    total["prompt"] += usage.prompt_tokens
    total["completion"] += usage.completion_tokens
    total["total"] += usage.total_tokens


def _critique(client, model_critic, question, observation) -> tuple[str, Any]:
    resp = client.chat.completions.create(
        model=model_critic,
        messages=[
            {"role": "system",
             "content": "Ты критик. Оцени результат инструмента по задаче. "
                        "Ответь строго словом OK либо строкой 'REVISE: <причина>'."},
            {"role": "user",
             "content": f"Задача: {question}\nРезультат инструмента: {observation}"},
        ],
    )
    return resp.choices[0].message.content, resp.usage


def run_react_with_reflection(
    question: str,
    tools: list[dict],
    tool_dispatch: dict[str, Callable[..., Any]],
    max_iterations: int = 10,
    timeout_per_iteration_sec: float = 10.0,
    max_revisions: int = 2,
    model_main: str = "deepseek-v4-flash",
    model_critique: str = "deepseek-v4-flash",
    model_premium: str = "deepseek-v4-pro",
    client: OpenAI | None = None,
) -> dict:
    client = client or OpenAI()

    system_prompt = (
        "Действуй как агент: самостоятельно выбирай инструменты и их порядок, "
        "при необходимости разбивая задачу на подзадачи. "
        "На каждом шаге сначала одним предложением поясни, что и зачем делается, "
        "затем вызови ровно один инструмент и опирайся на его результат. "
        "Как только данных достаточно – дай финальный ответ без вызова инструментов. "
        "Не выдумывай данные: используй только то, что вернули инструменты; "
        "если доступными инструментами задачу решить нельзя – прямо сообщи об этом."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    revisions_used = 0        # ДО цикла — не сбрасывается между итерациями
    raised_to_premium = False
    usage_total = {"prompt": 0, "completion": 0, "total": 0}
    trace = []

    for step in range(max_iterations):
        started = time.monotonic()

        # --- Action: основная модель + timeout ---
        try:
            response = client.chat.completions.create(
                model=model_premium if raised_to_premium else model_main,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                timeout=timeout_per_iteration_sec,
            )
        except APITimeoutError:
            logger.warning("step=%d превышен timeout=%ss", step, timeout_per_iteration_sec)
            return {"answer": "Timeout", "steps": step + 1,
                    "usage": usage_total, "trace": trace, "error": "timeout",
                    "revisions": revisions_used}
        except Exception as exc:
            logger.exception("step=%d ошибка LLM-вызова", step)
            return {"answer": f"Ошибка LLM: {exc}", "steps": step + 1,
                    "usage": usage_total, "trace": trace, "error": "llm_error",
                    "revisions": revisions_used}

        duration_ms = round((time.monotonic() - started) * 1000, 1)
        _accumulate_usage(usage_total, response.usage)

        msg = response.choices[0].message
        messages.append(msg)

        logger.info("step=%d tools=%s duration_ms=%s usage=%s",
                 step, [c.function.name for c in (msg.tool_calls or [])],
                 duration_ms, usage_total)

        # --- остановка: финальный ответ без tool_calls ---
        if not msg.tool_calls:
            return {"answer": msg.content, "steps": step + 1,
                    "usage": usage_total, "trace": trace,
                    "revisions": revisions_used}

        # --- Observation: выполняем tools ---
        for call in msg.tool_calls:
            name = call.function.name
            args = None
            duration_tool_exec_ms = 0.0
            try:
                args = json.loads(call.function.arguments or "{}")
                started_tool_exec = time.monotonic()
                result = tool_dispatch[name](**args)
                duration_tool_exec_ms = round((time.monotonic() - started_tool_exec) * 1000, 1)
                observation = json.dumps({"status": "ok", "result": result}, ensure_ascii=False)
            except Exception as exc:
                observation = json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False)

            messages.append({"role": "tool", "tool_call_id": call.id, "content": observation})
            trace.append({"step": step, "tool_name": name, "tool_args": args,
                          "tool_result": str(observation)[:200], "duration_ms": duration_tool_exec_ms})

            # --- Reflexion-light: критик на КАЖДЫЙ observation ---
            if revisions_used < max_revisions:
                critique, crit_usage = _critique(client, model_critique, question, observation)
                _accumulate_usage(usage_total, crit_usage)
                if str(critique).strip().startswith("REVISE"):
                    revisions_used += 1
                    raised_to_premium = True
                    messages.append({"role": "system", "content": f"Критика: {critique}"})
                    logger.info("revision=%d/%d critique=%s", revisions_used, max_revisions, critique)

    return {"answer": "Превышен лимит итераций", "steps": max_iterations,
            "usage": usage_total, "trace": trace, "error": "max_iterations",
            "revisions": revisions_used}


def get_provider_url_and_key(provider:str)-> tuple[str,str]:
    settings = get_settings()
    if provider == "deepseek":
        return settings.llm.deepseek_base_url, settings.llm.deepseek_api_key.get_secret_value()
    if provider == "openai":
        return settings.llm.openai_base_url, settings.llm.openai_api_key.get_secret_value()
    if provider == "openrouter":
        return settings.llm.openrouter_base_url, settings.llm.openrouter_api_key.get_secret_value()
    return settings.llm.ollama_base_url, "ollama"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Агент ReAct на Chat Completions")
    parser.add_argument("task", help="Задача для агента")
    parser.add_argument("--max_iterations", type=int, default=10, help="Лимит шагов (guardrail)")
    parser.add_argument("--timeout_per_iteration_sec", type=float, default=10.0, help="Таймаут (сек)")
    parser.add_argument("--max_revisions", type=int, default=2, help="Макс. кол-во ревизий")
    parser.add_argument("--provider", help="Провайдер Chat Completions")
    parser.add_argument("--model_main", type=str, default="deepseek-v4-flash", help="Основная модель")
    parser.add_argument("--model_critique", type=str, default="deepseek-v4-flash", help="Модель-критик")
    parser.add_argument("--model_premium", type=str, default="deepseek-v4-pro", help="Преимум модель")
    parser.add_argument("--trace", action="store_true", help="Печатать пошаговую трассу")
    parser.add_argument("--log-dir", default=None, help="Каталог логов (по умолчанию docs/agent-react-traces)")
    args = parser.parse_args(argv)

    settings = get_settings()

    model_main = args.model_main
    model_critique = args.model_critique
    model_premium = args.model_premium

    provider = args.provider
    if provider is None:
        provider = settings.llm.default_provider

    # --- Логи: свой файл на каждый прогон ---
    root = Path(__file__).resolve().parents[2]  # корень проекта IntelliBase
    log_dir = root / (args.log_dir or "docs/agent-react-traces")
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run-{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),  # консоль (как раньше)
            logging.FileHandler(log_path, encoding="utf-8"),  # файл
        ],
    )
    logger.info("task: %s", args.task)
    logger.info("provider=%s model_main=%s model_critique=%s model_premium=%s max_iterations=%s timeout_per_iteration_sec=%s max_revisions=%s",
                provider, model_main, model_critique, model_premium, args.max_iterations, args.timeout_per_iteration_sec, args.max_revisions)

    base_url,api_key = get_provider_url_and_key(provider)

    client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    result = run_react_with_reflection(
        question= args.task,
        tools=TOOLS,
        tool_dispatch=DISPATCH,
        max_iterations=args.max_iterations,
        timeout_per_iteration_sec=args.timeout_per_iteration_sec,
        max_revisions=args.max_revisions,
        model_main = args.model_main,
        model_critique=args.model_critique,
        model_premium=args.model_premium,
        client=client)

    if result.get("error"):
        logger.info("Остановка: %s (шагов: %s)", result["error"], result["steps"])
    else:
        logger.info("Ответ: %s", result["answer"])
        print(result["answer"])

    if args.trace:
        print("\n--- trace ---")
        trace_path = log_dir / f"run-{ts}.jsonl"
        with trace_path.open("w", encoding="utf-8") as f:
            for entry in result["trace"]:
                line = json.dumps(entry, ensure_ascii=False)
                print(line)
                f.write(line + "\n")
        logger.info("trace сохранён в %s", trace_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())