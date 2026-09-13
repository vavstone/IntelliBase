import pytest
import pytest_asyncio
from unittest.mock import AsyncMock
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.services.agent_persistent import build_agent
from app.tools.graph_tools import get_current_time

class FakeChat:
    """Content-aware заглушка: пока нет результата инструмента — просит send_telegram_message,
    после — финализирует. Устойчива к перезапуску узла при resume."""

    def bind_tools(self, tools):  # noqa: ANN001
        return self

    async def ainvoke(self, messages):  # noqa: ANN001
        if any(getattr(m, "type", "") == "tool" for m in messages):
            return AIMessage(content="Готово, сообщение обработано.", id="ai-final")
        return AIMessage(
            content="",
            id="ai-send",
            tool_calls=[
                {
                    "name": "send_telegram_message",
                    "args": {
                        "chat_id": "34564444",
                        "text": "Вот информация из базы знаний...",
                    },
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        )

@pytest_asyncio.fixture
async def graph():
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        await saver.setup()
        send_fn = AsyncMock()          # мок side-effect
        g = build_agent(saver, FakeChat(), [get_current_time], send_fn)
        yield g, send_fn               # вернуть и граф, и мок

def _initial() -> dict:
    return {"messages": [HumanMessage("отправь ...")], "iteration_count": 0,
            "tool_results": [], "draft": None, "sent": False}


@pytest.mark.asyncio
async def test_reaches_interrupt(graph):
    g, _ = graph
    result = await g.ainvoke(_initial(), {"configurable": {"thread_id": "t1"}})
    assert "__interrupt__" in result          # payload ушёл наружу
    snap = await g.aget_state({"configurable": {"thread_id": "t1"}})
    assert snap.next == ("confirm_and_send",)  # ждёт человека


@pytest.mark.asyncio
async def test_resume_true_sends(graph):
    g, send_fn = graph
    config = {"configurable": {"thread_id": "t2"}}
    await g.ainvoke(_initial(), config)                     # дошли до interrupt
    result = await g.ainvoke(Command(resume=True), config)  # одобрили
    assert result["sent"] is True
    send_fn.assert_called_once()          # side-effect выполнен ровно один раз


@pytest.mark.asyncio
async def test_resume_false_skips_side_effect(graph):
    g, send_fn = graph
    config = {"configurable": {"thread_id": "t3"}}
    await g.ainvoke(_initial(), config)
    result = await g.ainvoke(Command(resume=False), config)
    assert result["sent"] is False
    send_fn.assert_not_called()           # ← главное требование задания
