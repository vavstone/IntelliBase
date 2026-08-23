"""Прогон набора задач ДЗ 6.2 на выбранном агенте (naive или react).

Читает JSON с задачами, вызывает выбранного агента для каждой задачи по очереди,
печатает сводку и (опционально) дописывает нормализованные результаты в JSONL-файл.
Два прогона (naive + react) в один JSONL дают всё для сравнительного отчёта.

Использование:
    uv run python dev_tasks/run_agent_tasks.py --agent naive
    uv run python dev_tasks/run_agent_tasks.py --agent react --provider deepseek

Сбор результатов в один файл (для таблицы отчёта):
    uv run python dev_tasks/run_agent_tasks.py --agent naive --out results.jsonl
    uv run python dev_tasks/run_agent_tasks.py --agent react --out results.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASKS = ROOT / "dev_tasks" / "tasks-6-2.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Прогон задач ДЗ 6.2 на выбранном агенте")
    p.add_argument("--agent", choices=["naive", "react"], required=True,
                   help="какого агента вызывать")
    p.add_argument("--tasks", default=str(DEFAULT_TASKS),
                   help="путь к JSON с задачами")
    p.add_argument("--provider", default=None,
                   help="провайдер (по умолчанию из настроек)")
    p.add_argument("--model", default=None,
                   help="основная модель (для naive — model; для react — model_main)")
    p.add_argument("--max-steps", type=int, default=None,
                   help="naive: max_steps; react: max_iterations (None = дефолт агента)")
    p.add_argument("--timeout", type=float, default=10.0,
                   help="react: таймаут итерации, сек")
    p.add_argument("--max-revisions", type=int, default=2,
                   help="react: лимит ревизий")
    p.add_argument("--model-critique", default=None, help="react: модель-критик")
    p.add_argument("--model-premium", default=None, help="react: премиум-модель")
    p.add_argument("--out", default=None,
                   help="JSONL-файл для дописывания результатов")
    p.add_argument("--trace", action="store_true",
                   help="печатать полный trace каждой задачи")
    return p.parse_args()


def _load_tasks(path: Path) -> list[dict]:
    assert path.exists(), f"Не найден файл задач: {path}"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, list) and data, "tasks.json должен быть непустым списком"
    return data


def _tools_from_trace(trace: list[dict]) -> list[str]:
    return [e.get("tool_name") for e in (trace or []) if e.get("tool_name")]


def _naive_total_tokens(trace: list[dict]) -> int:
    """Сумма токенов по уникальным шагам (каждый шаг = один LLM-вызов).

    naive пишет в trace одну запись на каждый tool_call одного ответа, но
    input/output токены у них одинаковые — берём первое вхождение на шаг.
    """
    per_step: dict[int, int] = {}
    for e in trace or []:
        step = e.get("step")
        if step is not None and step not in per_step:
            per_step[step] = (e.get("llm_input_tokens") or 0) + (e.get("llm_output_tokens") or 0)
    return sum(per_step.values())


def _normalize(raw: dict, task: dict, agent: str) -> dict:
    trace = raw.get("trace", [])
    if agent == "react":
        usage = raw.get("usage", {})
        tokens_total = usage.get("total")
        revisions = raw.get("revisions")
    else:
        tokens_total = _naive_total_tokens(trace)
        revisions = None
    return {
        "id": task.get("id"),
        "category": task.get("category"),
        "question": task.get("question"),
        "expected_tools": task.get("expected_tools", []),
        "agent": agent,
        "steps": raw.get("steps"),
        "tokens_total": tokens_total,
        "tools": _tools_from_trace(trace),
        "revisions": revisions,
        "error": raw.get("error"),
        "answer": raw.get("answer"),
    }


def _print_task(norm: dict, idx: int, total: int, args: argparse.Namespace) -> None:
    print()
    print(f"=== Задача {norm['id']}/{total} [{norm['category']}] ===")
    print(f"вопрос    : {norm['question']}")
    print(f"шагов     : {norm['steps']}")
    print(f"токенов   : {norm['tokens_total']}")
    print(f"tools     : {norm['tools'] or '—'}  (ожидалось: {norm['expected_tools'] or '—'})")
    rev = "—" if norm["revisions"] is None else norm["revisions"]
    print(f"ревизий   : {rev}")
    print(f"error     : {norm['error'] or '—'}")
    answer = norm["answer"]
    answer = "—" if answer is None else str(answer)
    print(f"ответ     : {answer[:150]}")
    if args.trace:
        print(f"trace     : {json.dumps(norm['tools'], ensure_ascii=False)}")


def _print_summary(results: list[dict], agent: str) -> None:
    print()
    print("=" * 70)
    print(f"СВОДКА (agent={agent})")
    print("=" * 70)
    print(f"{'#':<3}{'категория':<16}{'шагов':<7}{'токенов':<9}{'ревизий':<9}error")
    for r in results:
        rev = "—" if r["revisions"] is None else str(r["revisions"])
        print(f"{r['id']:<3}{r['category']:<16}{str(r['steps']):<7}"
              f"{str(r['tokens_total']):<9}{rev:<9}{r['error'] or '—'}")


def _append_jsonl(out: str, norm: dict) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(norm, ensure_ascii=False) + "\n")


def main() -> int:
    args = _parse_args()
    tasks = _load_tasks(Path(args.tasks))

    from openai import OpenAI
    from app.core.config import get_settings

    settings = get_settings()
    provider = args.provider or settings.llm.default_provider

    if args.agent == "naive":
        from app.services.agent_naive import get_provider_url_and_key, run_agent

        base_url, api_key = get_provider_url_and_key(provider)
        client = OpenAI(base_url=base_url, api_key=api_key)
        model = args.model or settings.llm.default_model

        def call(question: str) -> dict:
            kw: dict = {"task": question, "model": model, "client": client}
            if args.max_steps is not None:
                kw["max_steps"] = args.max_steps
            return run_agent(**kw)

    else:
        from app.services.agent_react import get_provider_url_and_key, run_react_with_reflection
        from app.tools.react_tools import TOOLS, DISPATCH

        base_url, api_key = get_provider_url_and_key(provider)
        client = OpenAI(base_url=base_url, api_key=api_key)
        model_main = args.model or "deepseek-v4-flash"
        model_critique = args.model_critique or "deepseek-v4-flash"
        model_premium = args.model_premium or "deepseek-v4-pro"

        def call(question: str) -> dict:
            kw = dict(
                question=question,
                tools=TOOLS,
                tool_dispatch=DISPATCH,
                client=client,
                timeout_per_iteration_sec=args.timeout,
                max_revisions=args.max_revisions,
                model_main=model_main,
                model_critique=model_critique,
                model_premium=model_premium,
            )
            if args.max_steps is not None:
                kw["max_iterations"] = args.max_steps
            return run_react_with_reflection(**kw)

    results: list[dict] = []
    total = len(tasks)
    for i, task in enumerate(tasks):
        raw = call(task["question"])
        norm = _normalize(raw, task, args.agent)
        results.append(norm)
        _print_task(norm, i + 1, total, args)
        if args.out:
            _append_jsonl(args.out, norm)

    _print_summary(results, args.agent)
    if args.out:
        print(f"\nРезультаты дописаны в {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
