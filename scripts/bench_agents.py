import argparse
import asyncio
import json
import time
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from openai import OpenAI

from app.services.agent_graph import custom_graph, prebuilt_graph
from app.core.config import get_settings
from app.services.agent_react import get_provider_url_and_key, run_react_with_reflection
from app.tools.react_tools import TOOLS as REACT_TOOLS, DISPATCH


def _usage_from_messages(messages) -> dict:
    prompt = completion = 0
    for m in messages:
        if isinstance(m, AIMessage) and m.usage_metadata:
            prompt += m.usage_metadata.get("input_tokens", 0)
            completion += m.usage_metadata.get("output_tokens", 0)
    return {"prompt": prompt, "completion": completion}


async def run_custom(q: str, task_id: int) -> dict:
    state = {"messages": [HumanMessage(content=q)], "iteration_count": 0, "tool_results": []}
    config = {"configurable": {"thread_id": f"bench-{task_id}"}}   # шаг 10 (бонус)
    t0 = time.perf_counter()
    final = await custom_graph.ainvoke(state, config=config)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    usage = _usage_from_messages(final["messages"])
    tools_used = [m.name for m in final["messages"] if isinstance(m, ToolMessage)]
    return {
        "latency_ms": latency_ms,
        "steps": final["iteration_count"],
        **usage,
        "tools": tools_used,
        "answer": str(final["messages"][-1].content),
    }

async def run_prebuilt(q: str, task_id: int) -> dict:
    t0 = time.perf_counter()
    final = await prebuilt_graph.ainvoke({"messages": [HumanMessage(content=q)]})
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    usage = _usage_from_messages(final["messages"])
    tools_used = [m.name for m in final["messages"] if isinstance(m, ToolMessage)]
    return {
        "latency_ms": latency_ms,
        "steps": sum(1 for m in final["messages"] if isinstance(m, AIMessage)),
        **usage,
        "tools": tools_used,
        "answer": str(final["messages"][-1].content),
    }


async def run_react(q: str, client, model_main, react_tools, dispatch) -> dict:
    t0 = time.perf_counter()
    result = await asyncio.to_thread(
        run_react_with_reflection,
        question=q, tools=react_tools, tool_dispatch=dispatch,
        client=client, model_main=model_main,
        # timeout_per_iteration_sec=..., max_revisions=... — оставьте дефолты или задайте
    )
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "latency_ms": latency_ms,
        "steps": result["steps"],
        "prompt": result["usage"]["prompt"],
        "completion": result["usage"]["completion"],
        "tools": [e["tool_name"] for e in result.get("trace", []) if e.get("tool_name")],
        "answer": result.get("answer"),
    }


LOG = None  # открытый файл лога (None = не пишем)

def log(msg: str) -> None:
    print(msg, flush=True)
    if LOG is not None:
        LOG.write(msg + "\n")
        LOG.flush()


def _tools_str(tools: list) -> str:
    """Склеивает имена инструментов, игнорируя None/пустые записи трассы."""
    return ",".join(str(t) for t in tools if t) or "-"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Бенчмарк агентов 6.3: react vs custom vs prebuilt")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--out", default=None, help="JSONL-файл для сырых результатов")
    parser.add_argument("--log", default=None, help="файл для лога прогресса")
    parser.add_argument("--limit", type=int, default=0,
                        help="прогнать только первые N задач (0 = все)")

    global LOG
    args = parser.parse_args()

    if args.log:
        LOG = open(args.log, "w", encoding="utf-8")  # 'a' — если дописывать

    settings = get_settings()
    provider = args.provider or settings.llm.default_provider
    base_url, api_key = get_provider_url_and_key(provider)
    client = OpenAI(base_url=base_url, api_key=api_key)
    model_main = args.model or "deepseek-v4-flash"

    tasks_path = Path(__file__).resolve().parents[1] / "dev_tasks" / "tasks-6-2.json"
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    if args.limit:
        tasks = tasks[: args.limit]

    IMPLS = ["react", "custom", "prebuilt"]
    rows = []  # сюда складывать строки для таблицы

    for task in tasks:
        q, tid = task["question"], task["id"]
        for impl in IMPLS:
            latencies = []
            last = None
            for _ in range(args.runs):
                log(f"  [{task['id']}/{len(tasks)}] {impl} run {_+1}/{args.runs}: {q[:40]}")
                if impl == "react":
                    r = await run_react(q, client, model_main, REACT_TOOLS, DISPATCH)
                elif impl == "custom":
                    r = await run_custom(q, tid)
                else:
                    r = await run_prebuilt(q, tid)
                latencies.append(r["latency_ms"])
                last = r
                await asyncio.sleep(args.sleep)  # пауза против rate-limit
            avg_ms = round(sum(latencies) / len(latencies), 1)
            rows.append((task, impl, avg_ms, last))

    header = (
        f"{'#':<3}{'задача':<33}{'реализация':<10}{'latency_ms':<12}{'prompt':<8}"
        f"{'completion':<12}{'steps':<6}{'tools':<22}{'answer'}")
    print(header)
    if LOG is not None:
        LOG.write(header + "\n")

    for task, impl, avg_ms, last in rows:
        answer = str(last["answer"])[:40] if last["answer"] else "—"
        line = (
            f"{task['id']:<3}{task['question'][:31]:<33}{impl:<10}{avg_ms:<12}{last['prompt']:<8}"
            f"{last['completion']:<12}{last['steps']:<6}{_tools_str(last['tools']):<22}{answer}")
        print(line)
        if LOG is not None:
            LOG.write(line + "\n")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for task, impl, avg_ms, last in rows:
                f.write(json.dumps({
                    "id": task["id"],
                    "question": task["question"],
                    "impl": impl,
                    "latency_ms": avg_ms,
                    "prompt": last["prompt"],
                    "completion": last["completion"],
                    "steps": last["steps"],
                    "tools": _tools_str(last["tools"]),
                    "answer": last["answer"],
                }, ensure_ascii=False) + "\n")
        log(f"Результаты записаны в {args.out}")

    if LOG is not None:
        LOG.close()


if __name__ == "__main__":
    asyncio.run(main())