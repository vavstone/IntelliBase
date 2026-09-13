"""Персистентный ReAct-агент: чекпоинтер + человек в цикле на опасном действии.

Инкремент к базовому ReAct-агенту: тот же цикл, но граф компилируется с
чекпоинтером (состояние переживает рестарт), а опасный инструмент `send_telegram_message`
проходит через человека — два узла:

- `prepare_send` — idempotent: рендерит payload сообщения из tool_call, без side-effect;
- `confirm_and_send` — `interrupt()` перед отправкой, реальная отправка ТОЛЬКО после
  `Command(resume=...)`. При роли `full` interrupt пропускается (политика доступа).

Бэкенд чекпоинтера выбирается через `AGENT_CHECKPOINTER`: `memory` | `sqlite` |
`postgres`. Схему чекпоинтера ведёт `setup()`, доменную — Alembic (в `env.py`
таблицы `checkpoint*` исключены из autogenerate через `include_name`).
"""
import operator
from contextlib import asynccontextmanager
from typing import Literal, TypedDict, Annotated, Callable, Awaitable, Any, AsyncIterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage, AnyMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt

from app.tools.graph_tools import TOOLS
from app.core.config import get_settings

MAX_ITERATIONS = 6
DANGEROUS_TOOL = "send_telegram_message"

# Реальный side-effect отправки: async-callable, инжектируется в фабрику, чтобы
# в тестах подменяться моком и вызываться ТОЛЬКО после одобрения человеком.
SendTelegramFn = Callable[[dict], Awaitable[None]]

class PersistentAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    iteration_count: int
    tool_results: Annotated[list[dict], operator.add]
    draft: dict | None  # payload сообщения, подготовленный prepare_send (до отправки)
    sent: bool  # выполнен ли side-effect отправки

def _find_call(message: AnyMessage, name: str) -> dict:
    for call in message.tool_calls:
        if call["name"] == name:
            return call
    raise ValueError(f"в сообщении нет tool_call {name!r}")


def build_agent(
    checkpointer: Any,
    model: BaseChatModel,
    tools: list[BaseTool],
    send_telegram_fn: SendTelegramFn):
    """Компилирует персистентный ReAct-граф с HIL-гейтом на `send_telegram_message`.
    `tools` — безопасные инструменты (get_current_time, search_knowledge_base). Опасный
    `send_telegram_message` добавляется здесь и исполняется не в `execute_tool`, а через
    отдельную ветку с `interrupt`."""
    bound_model = model.bind_tools(tools)
    tool_by_name = {t.name: t for t in tools}

    async def call_model(state: PersistentAgentState) -> dict:
        response = await bound_model.ainvoke(state["messages"])
        return {
            "messages": [response],
            "iteration_count": state["iteration_count"] + 1,
        }

    async def execute_tool(state: PersistentAgentState) -> dict:
        last = state["messages"][-1]
        messages: list = []
        results: list[dict] = []
        for call in last.tool_calls:
            if call["name"] == DANGEROUS_TOOL:
                continue  # опасный инструмент идёт через HIL-ветку, не здесь
            if call["name"] not in tool_by_name:
                content = f"error: unknown tool '{call['name']}'"
            else:
                content = str(await tool_by_name[call["name"]].ainvoke(call["args"]))
            messages.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
            results.append(
                {"name": call["name"], "args": call["args"], "result": content}
            )
        return {"messages": messages, "tool_results": results}

    async def prepare_send(state: PersistentAgentState) -> dict:
        """Idempotent: собирает payload сообщения из tool_call. Без side-effect."""
        call = _find_call(state["messages"][-1], DANGEROUS_TOOL)
        args = call["args"]
        draft = {
            "chat_id": args.get("chat_id", ""),
            "text": args.get("text", ""),
            "tool_call_id": call["id"],
        }
        return {"draft": draft}

    async def confirm_and_send(
        state: PersistentAgentState, config: RunnableConfig
    ) -> dict:
        """interrupt перед отправкой; реальная отправка — ПОСЛЕ resume."""
        draft = state["draft"] or {}
        role = (config.get("configurable") or {}).get("user_role", "write-with-approve")
        if role == "full":
            decision: Any = True  # полный доступ — без подтверждения человека
        else:
            decision = interrupt({"type": "approve_send", "preview": draft})
        approved = decision is True or decision == "approve"
        if approved:
            await send_telegram_fn(draft)  # SIDE-EFFECT только здесь, после resume
            content = f"сообщение отправлено: {draft.get('text', '')}"
        else:
            content = "отправка отменена пользователем"
        return {
            "sent": approved,
            "messages": [
                ToolMessage(content=content, tool_call_id=draft.get("tool_call_id", ""))
            ],
            "tool_results": [
                {"name": DANGEROUS_TOOL, "args": draft, "result": content}
            ],
        }
	
    async def force_finish(state: PersistentAgentState) -> dict:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return {"messages": [AIMessage(content="Превышен лимит итераций")]}
        return {}

    def route_after_model(
        state: PersistentAgentState,
    ) -> Literal["execute_tool", "prepare_send", "force_finish"]:
        if state["iteration_count"] >= MAX_ITERATIONS:
            return "force_finish"
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None)
        if not calls:
            return "force_finish"
        if any(call["name"] == DANGEROUS_TOOL for call in calls):
            return "prepare_send"
        return "execute_tool"

    builder = StateGraph(PersistentAgentState)
    builder.add_node("call_model", call_model)
    builder.add_node("execute_tool", execute_tool)
    builder.add_node("prepare_send", prepare_send)
    builder.add_node("confirm_and_send", confirm_and_send)
    builder.add_node("force_finish", force_finish)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        route_after_model,
        {
            "execute_tool": "execute_tool",
            "prepare_send": "prepare_send",
            "force_finish": "force_finish",
        },
    )
    builder.add_edge("execute_tool", "call_model")
    builder.add_edge("prepare_send", "confirm_and_send")
    builder.add_edge("confirm_and_send", "call_model")
    builder.add_edge("force_finish", END)
    return builder.compile(checkpointer=checkpointer)


@asynccontextmanager
async def agent_lifespan(
    backend: Literal["memory", "sqlite", "postgres"],
    model: BaseChatModel,
    tools: list[BaseTool],
    send_telegram_fn: SendTelegramFn,
    *,
    sqlite_path: str = "var/agent_checkpoints.sqlite",
    postgres_url: str = "",
) -> AsyncIterator[Any]:
    """Поднимает нужный чекпоинтер и отдаёт скомпилированный граф.
    `setup()` вызывается ровно один раз здесь — не на каждый запрос.
    """
    if backend == "memory":
        # InMemorySaver не требует setup() и живёт в памяти процесса.
        yield build_agent(InMemorySaver(), model, tools, send_telegram_fn)
    elif backend == "sqlite":
        from pathlib import Path

        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(sqlite_path) as saver:
            await saver.setup()
            yield build_agent(saver, model, tools, send_telegram_fn)
    elif backend == "postgres":
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        async with AsyncPostgresSaver.from_conn_string(
            postgres_url
        ) as saver:
            await saver.setup()
            yield build_agent(saver, model, tools, send_telegram_fn)
    else:
        raise ValueError(f"неизвестный AGENT_CHECKPOINTER: {backend!r}")