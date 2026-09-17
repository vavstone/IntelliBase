import json
import time
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from experiments.metrics_callback import MetricsCallback

TEAM = {"researcher", "writer"}

def count_handoffs(nodes: list[str]) -> int:
    """Передача управления = новый непрерывный блок узлов агента команды.
    supervisor -> researcher = 1; ... -> supervisor -> writer = ещё 1.
    У single-agent таких узлов нет -> 0."""
    handoffs, prev_in_team = 0, False
    for node in nodes:
        in_team = node in TEAM
        if in_team and not prev_in_team:
            handoffs += 1
        prev_in_team = in_team
    return handoffs


async def run_one(app, question: dict, impl: str) -> dict:
    metrics = MetricsCallback()
    config = {"configurable": {"thread_id": f"exp-langgraph-{question['id']}"},
              "callbacks": [metrics]}

    nodes: list[str] = []
    last_ai = ""
    last_agent_ai = ""
    t0 = time.perf_counter()
    async for chunk in app.astream({"messages": [HumanMessage(content=question["text"])]},
                                   config, stream_mode="updates"):
        for node, update in chunk.items():
            nodes.append(node)
            print(f"--- {node} ---")
            msgs = update.get("messages") or []
            # ответом считаем только сообщение модели: в узле tools последним идёт
            # ToolMessage с JSON от RAG, его в answer брать нельзя
            if msgs and isinstance(msgs[-1], AIMessage) and str(msgs[-1].content):
                content = str(msgs[-1].content)
                last_ai = content
                if node in TEAM:
                    # в мультиагенте ответ пользователю — текст агента команды (writer),
                    # а не служебная реплика супервизора
                    last_agent_ai = content
                print(content[:200])
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {"impl": impl, "qid": question["id"], "kind": question["kind"],
            "total_tokens": metrics.total_tokens, "llm_calls": metrics.llm_calls,
            "latency_ms": latency_ms, "handoff_count": count_handoffs(nodes),
            "answer": last_agent_ai or last_ai}


def save_results(records: list[dict], path: Path) -> None:
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    merged = {(r["impl"], r["qid"]): r for r in existing}
    merged.update({(r["impl"], r["qid"]): r for r in records})
    path.write_text(json.dumps(list(merged.values()), ensure_ascii=False, indent=2),
                    encoding="utf-8")
