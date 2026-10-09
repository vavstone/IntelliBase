"""Юнит-тесты роутера агента: сборка config и начального состояния (app/routers/agent.py).

Проверяется контракт с ботом: получателя выбирает сервер (config), а служебная
подсказка о чате-инициаторе добавляется здесь, а не клиентом — иначе вызовы API
напрямую (curl, будущий UI) не знают, кто спрашивает.
"""

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage

from app.main import app
from app.routers.agent import _config, _initial_state

INITIATOR = "100000001"


def _text(state: dict) -> str:
    return state["messages"][0].content


def test_initiator_hint_is_added_by_server() -> None:
    state = _initial_state("отправь мне отчёт", INITIATOR)
    text = _text(state)
    assert "отправь мне отчёт" in text
    assert INITIATOR in text
    assert "find_recipient" in text
    # Чат инициатора НЕ объявляется получателем: иначе задача «отправь Иванову»
    # получает двух разных адресатов и агент останавливается.
    assert "получателя:" not in text.lower()


def test_message_without_initiator_is_untouched() -> None:
    assert _text(_initial_state("отправь отчёт")) == "отправь отчёт"
    assert _text(_initial_state("отправь отчёт", None)) == "отправь отчёт"


def test_state_starts_with_empty_draft() -> None:
    state = _initial_state("задача", INITIATOR)
    assert state["iteration_count"] == 0
    assert state["tool_results"] == []
    assert state["draft"] is None
    assert state["sent"] is False


def test_config_carries_recipients_and_initiator() -> None:
    config = _config(
        "thread-1",
        delivery_chat_id=INITIATOR,
        allowed_recipients={"111222333": "Иванов Пётр"},
    )
    conf = config["configurable"]
    assert conf["thread_id"] == "thread-1"
    assert conf["delivery_chat_id"] == INITIATOR
    assert conf["allowed_recipients"] == {"111222333": "Иванов Пётр"}


def test_config_defaults_are_safe() -> None:
    """Без явных получателей конфиг не разрешает никого (кроме инициатора)."""
    conf = _config("thread-2")["configurable"]
    assert conf["allowed_recipients"] == {}
    assert conf["delivery_chat_id"] is None


# --- HIL: отсутствие decision — не согласие ------------------------------


def test_resume_request_has_no_fail_open_default() -> None:
    """Поле decision не подставляется по умолчанию: молчание — не «да»."""
    from app.routers.agent import AgentResumeRequest

    assert AgentResumeRequest(thread_id="t").decision is None


@pytest.mark.asyncio
async def test_resume_without_decision_is_rejected() -> None:
    """POST /agent/resume без decision → 422, до графа запрос не доходит."""
    from fastapi import HTTPException

    from app.routers.agent import AgentResumeRequest, agent_resume

    with pytest.raises(HTTPException) as exc:
        await agent_resume(
            AgentResumeRequest(thread_id="t"), graph=object(), session_factory=None
        )
    assert exc.value.status_code == 422
    assert "decision" in exc.value.detail


# --- HTTP-слой: маппинг статусов графа и гейты ---------------------------


class _FakeGraph:
    """Заглушка скомпилированного графа: возвращает заданный результат."""

    def __init__(self, result: dict) -> None:
        self.result = result
        self.calls: list[tuple] = []

    async def ainvoke(self, payload, config=None):  # noqa: ANN001
        self.calls.append((payload, config))
        return self.result


@pytest.fixture
def _agent_state():
    """app.state.agent_graph управляется тестом и восстанавливается после."""
    had = hasattr(app.state, "agent_graph")
    prev = getattr(app.state, "agent_graph", None)
    yield
    if had:
        app.state.agent_graph = prev
    else:
        try:
            del app.state.agent_graph
        except AttributeError:
            pass


async def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_chat_done_maps_to_answer(_agent_state) -> None:
    app.state.agent_graph = _FakeGraph(
        {"messages": [AIMessage(content="итоговый ответ")], "tool_results": []}
    )
    async with await _client() as ac:
        resp = await ac.post(
            "/agent/chat", json={"message": "привет", "chat_id": INITIATOR}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "done"
    assert body["answer"] == "итоговый ответ"
    # Инициатор доезжает до config — получателя выбирает сервер, не модель.
    _, config = app.state.agent_graph.calls[0]
    assert config["configurable"]["delivery_chat_id"] == INITIATOR


@pytest.mark.asyncio
async def test_chat_interrupted_returns_payload(_agent_state) -> None:
    from types import SimpleNamespace

    preview = {"kind": "message", "chat_ids": [INITIATOR], "recipients": []}
    app.state.agent_graph = _FakeGraph(
        {
            "__interrupt__": [
                SimpleNamespace(value={"type": "approve_send", "preview": preview})
            ],
            "tool_results": [],
        }
    )
    async with await _client() as ac:
        resp = await ac.post(
            "/agent/chat", json={"message": "отправь", "chat_id": INITIATOR}
        )
    body = resp.json()
    assert body["status"] == "interrupted"
    assert body["interrupt"]["type"] == "approve_send"
    assert body["interrupt"]["preview"] == preview


@pytest.mark.asyncio
async def test_chat_503_without_graph(_agent_state) -> None:
    app.state.agent_graph = None
    async with await _client() as ac:
        resp = await ac.post("/agent/chat", json={"message": "привет"})
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_resume_passes_decision_to_graph(_agent_state) -> None:
    """Решение человека уходит в граф как Command(resume=...) того же треда."""
    from langgraph.types import Command

    app.state.agent_graph = _FakeGraph(
        {"messages": [AIMessage(content="отправлено")], "tool_results": []}
    )
    async with await _client() as ac:
        resp = await ac.post(
            "/agent/resume",
            json={"thread_id": "t-http", "decision": False, "chat_id": INITIATOR},
        )
    assert resp.status_code == 200
    payload, config = app.state.agent_graph.calls[0]
    assert isinstance(payload, Command)
    assert getattr(payload, "resume", object()) is False
    assert config["configurable"]["thread_id"] == "t-http"
