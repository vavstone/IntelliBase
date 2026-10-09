"""Ручки агентного слоя: прогон персистентного ReAct-графа с HIL.

- `POST /agent/chat` — один шаг диалога; если агент дошёл до опасного действия,
  вернётся `status="interrupted"` с payload для подтверждения.
- `POST /agent/resume` — возобновление после подтверждения человеком.
- `POST /agent/stream` — SSE-поток прогресса по узлам и токенов LLM.
"""

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel

from app.core.config import get_settings
from app.deps.providers import AgentGraphDep, SessionFactoryDep
from app.services.bot_users import load_active_users, resolve_allowed

router = APIRouter(prefix="/agent", tags=["agent"])


async def _allowed_recipients(session_factory) -> dict[str, str]:
    """{chat_id: имя} — кому разрешена отправка: пользователи бота из БД + bootstrap.

    Считается на каждый запрос (и на resume тоже): отозванный доступ перестаёт
    действовать сразу, без перезапуска сервиса.
    """
    rows = await load_active_users(session_factory)
    return resolve_allowed(rows, get_settings().bot_allowed_chat_ids)


def _config(
    thread_id: str,
    user_role: str = "write-with-approve",
    delivery_chat_id: str | None = None,
    allowed_recipients: dict[str, str] | None = None,
) -> dict:
    """Конфиг графа.

    `delivery_chat_id` — инициатор запроса, `allowed_recipients` — список
    пользователей бота с именами. Оба передаются в config, а не в тексте задачи:
    адресата выбирает сервер, не LLM, иначе любой пользователь бота мог бы
    попросить отправить данные на указанный им chat_id.
    """
    return {
        "configurable": {
            "thread_id": thread_id,
            "user_role": user_role,
            "delivery_chat_id": delivery_chat_id,
            "allowed_recipients": allowed_recipients or {},
        }
    }


def _initial_state(message: str, initiator_chat_id: str | None = None) -> dict:
    """Начальное состояние графа.

    Служебная подсказка о чате-инициаторе добавляется здесь, а не в боте: она
    нужна любому клиенту (бот, curl, будущий UI), чтобы модель понимала «отправь
    мне» и при этом не выдумывала chat_id. Получателем этот чат при этом не
    объявляется — иначе задача «отправь Иванову» получает двух разных адресатов.
    """
    text = message
    if initiator_chat_id:
        text = (
            f"{message}\n\n"
            f"(Служебно: задача пришла из чата {initiator_chat_id}. "
            f"Нужного получателя ищи через find_recipient.)"
        )
    return {
        "messages": [HumanMessage(text)],
        "iteration_count": 0,
        "tool_results": [],
        "draft": None,
        "sent": False,
    }


class AgentChatRequest(BaseModel):
    message: str
    thread_id: str = "default"
    # Чат, из которого пришёл запрос (Telegram chat_id). Он же — разрешённый
    # получатель отправки: без него инструмент отправки отклонит любой адрес.
    chat_id: str | None = None


class AgentChatResponse(BaseModel):
    status: str  # "done" | "interrupted"
    thread_id: str
    answer: str | None = None
    tool_results: list[dict] = []
    interrupt: dict | None = None


def _to_response(result: dict, thread_id: str) -> AgentChatResponse:
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        return AgentChatResponse(
            status="interrupted",
            thread_id=thread_id,
            interrupt=payload,
            tool_results=result.get("tool_results", []),
        )
    final = result["messages"][-1]
    return AgentChatResponse(
        status="done",
        thread_id=thread_id,
        answer=final.content or "",
        tool_results=result.get("tool_results", []),
    )


@router.post("/chat", response_model=AgentChatResponse)
async def agent_chat(
    req: AgentChatRequest, graph: AgentGraphDep, session_factory: SessionFactoryDep
) -> AgentChatResponse:
    if graph is None:
        raise HTTPException(status_code=503, detail="агентный граф не инициализирован")
    config = _config(
        req.thread_id,
        delivery_chat_id=req.chat_id,
        allowed_recipients=await _allowed_recipients(session_factory),
    )
    result = await graph.ainvoke(
        _initial_state(req.message, req.chat_id), config
    )
    return _to_response(result, req.thread_id)


class AgentResumeRequest(BaseModel):
    thread_id: str
    # Решение обязательно, без дефолта: на опасном действии «поле не передали»
    # должно означать отказ запроса, а не согласие. Раньше дефолт был True —
    # POST без поля молча подтверждал отправку (fail-open на HIL).
    decision: bool | str | None = None
    # Тот же инициатор, что и в /agent/chat: узел отправки перезапускается на
    # resume с конфигом текущего вызова, поэтому chat_id нужен и здесь.
    chat_id: str | None = None


@router.post("/resume", response_model=AgentChatResponse)
async def agent_resume(
    req: AgentResumeRequest, graph: AgentGraphDep, session_factory: SessionFactoryDep
) -> AgentChatResponse:
    if graph is None:
        raise HTTPException(status_code=503, detail="агентный граф не инициализирован")
    if req.decision is None:
        raise HTTPException(
            status_code=422,
            detail="decision обязателен: true/\"approve\" — подтвердить, false/\"reject\" — отменить",
        )
    config = _config(
        req.thread_id,
        delivery_chat_id=req.chat_id,
        allowed_recipients=await _allowed_recipients(session_factory),
    )
    result = await graph.ainvoke(Command(resume=req.decision), config)
    return _to_response(result, req.thread_id)


class AgentStreamRequest(BaseModel):
    thread_id: str
    input: dict | None = None  # старт: {"messages": [...]}
    resume: bool | str | None = None  # возобновление после interrupt
    chat_id: str | None = None  # инициатор запроса = разрешённый получатель


def _format_event(stream_type: str, payload: Any) -> dict | None:
    if stream_type == "updates":
        if isinstance(payload, dict) and "__interrupt__" in payload:
            interrupts = payload["__interrupt__"]
            value = interrupts[0].value if interrupts else {}
            return {"type": "interrupt", "payload": value}
        return {"type": "update", "nodes": list(payload.keys())}
    if stream_type == "messages":
        chunk, _meta = payload
        text = getattr(chunk, "content", "")
        return {"type": "token", "text": text} if text else None
    return None


@router.post("/stream")
async def agent_stream(
    req: AgentStreamRequest, graph: AgentGraphDep, session_factory: SessionFactoryDep
) -> StreamingResponse:
    if graph is None:
        raise HTTPException(status_code=503, detail="агентный граф не инициализирован")

    if req.resume is not None:
        graph_input: Any = Command(resume=req.resume)
    elif req.input is not None:
        graph_input = {
            "iteration_count": 0,
            "tool_results": [],
            "draft": None,
            "sent": False,
            **req.input,
        }
    else:
        raise HTTPException(status_code=422, detail="нужен input или resume")

    config = _config(
        req.thread_id,
        delivery_chat_id=req.chat_id,
        allowed_recipients=await _allowed_recipients(session_factory),
    )

    async def event_source() -> AsyncIterator[str]:
        async for stream_type, payload in graph.astream(
            graph_input, config, stream_mode=["updates", "messages"]
        ):
            event = _format_event(stream_type, payload)
            if event is not None:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(event_source(), media_type="text/event-stream")
