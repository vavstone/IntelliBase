"""Персистентный ReAct-агент: чекпоинтер + человек в цикле на опасном действии.

Инкремент к базовому ReAct-агенту: тот же цикл, но граф компилируется с
чекпоинтером (состояние переживает рестарт), а опасный инструмент `send_telegram_message`
проходит через человека — два узла:

- `prepare_send` — idempotent: рендерит payload сообщения из tool_call, без side-effect;
- `confirm_and_send` — `interrupt()` перед отправкой, реальная отправка ТОЛЬКО после
  `Command(resume=...)`. При роли `full` interrupt пропускается (политика доступа).

Получателя выбирает не модель: узел `route_after_model` сверяет chat_id из tool_call
с множеством разрешённых — список пользователей бота из БД плюс чат-инициатор
запроса (`allowed_recipients` в config) — и при промахе отправляет вызов в
`reject_send`: агент получает ошибку инструмента и переспрашивает, отправки наружу
не происходит. Без этой проверки адресата выбирал бы текст задачи, прошедший
через LLM.

Бэкенд чекпоинтера выбирается через `AGENT_CHECKPOINTER`: `memory` | `sqlite` |
`postgres`. Схему чекпоинтера ведёт `setup()`, доменную — Alembic (в `env.py`
таблицы `checkpoint*` исключены из autogenerate через `include_name`).
"""
import logging
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

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 6

# Опасные инструменты: единственные с побочным эффектом наружу. Граф их не
# исполняет — ведёт через проверку получателя и подтверждение человеком.
SEND_MESSAGE_TOOL = "send_telegram_message"
SEND_DOCUMENT_TOOL = "send_telegram_document"
DANGEROUS_TOOLS = (SEND_MESSAGE_TOOL, SEND_DOCUMENT_TOOL)
DANGEROUS_TOOL = SEND_MESSAGE_TOOL  # для обратной совместимости импортов

# Реальный side-effect отправки: async-callable, инжектируется в фабрику, чтобы
# в тестах подменяться моком и вызываться ТОЛЬКО после одобрения человеком.
# Возвращает человекочитаемый результат доставки (он уходит в ToolMessage);
# при неудаче бросает исключение — «отправлено» не должно быть ложью.
SendTelegramFn = Callable[[dict], Awaitable[str]]

# Резолвер файла корпуса: имя из sources[].file_name → путь (None — нет такого).
# Инжектируется, чтобы граф не знал про каталог корпуса и легко тестировался.
ResolveDocumentFn = Callable[[str], Any]
SendTelegramFn = Callable[[dict], Awaitable[str]]


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


def _find_dangerous_call(message: AnyMessage) -> dict | None:
    """Опасный вызов в сообщении (отправка текста или документа) — либо None."""
    for call in getattr(message, "tool_calls", None) or []:
        if call["name"] in DANGEROUS_TOOLS:
            return call
    return None


def _configurable(config: RunnableConfig | None) -> dict:
    return (config or {}).get("configurable") or {}


def allowed_recipients(config: RunnableConfig | None) -> dict[str, str]:
    """{chat_id: имя} — кому агенту разрешено отправлять в этом запросе.

    Источник — `config["configurable"]["allowed_recipients"]`: его считает роут
    (`app/routers/agent.py`) из списка пользователей бота в БД плюс bootstrap.
    Чат-инициатор добавляется всегда: он известен серверу, не зависит от текста
    задачи и от доступности БД — «отправь мне» работает даже при упавшем Postgres.
    """
    conf = _configurable(config)
    mapping = {
        str(chat_id): str(title or "")
        for chat_id, title in (conf.get("allowed_recipients") or {}).items()
    }
    initiator = str(conf.get("delivery_chat_id") or "").strip()
    if initiator:
        mapping.setdefault(initiator, "")
    return mapping


def permitted_chat_ids(config: RunnableConfig | None) -> set[str]:
    """chat_id, которым разрешена отправка (ключи `allowed_recipients`)."""
    return set(allowed_recipients(config))


def requested_recipient(message: AnyMessage) -> str:
    """Первый chat_id из tool_call — для сообщений об отказе (ещё не проверенный)."""
    call = _find_dangerous_call(message)
    if call is None:
        return ""
    return next(iter(requested_recipients(call)), "")


def requested_recipients(call: dict) -> list[str]:
    """chat_id, которые предложила модель в опасном tool_call.

    Модель передаёт список (`chat_ids`): одна отправка может адресоваться
    нескольким получателям («мне и Иванову») — подтверждение и проверка прав
    тогда одни на всех. Принимаем и старый одиночный `chat_id`.
    """
    args = call.get("args") or {}
    raw = args.get("chat_ids")
    if raw is None:
        raw = [args.get("chat_id")] if args.get("chat_id") else []
    if isinstance(raw, str):
        raw = [raw]
    return [str(item).strip() for item in raw if str(item or "").strip()]


def rejection_reason(
    message: AnyMessage,
    config: RunnableConfig | None,
    resolve_document: ResolveDocumentFn | None = None,
) -> str | None:
    """Почему опасный вызов нельзя исполнять. None — можно (уходит на подтверждение).

    Проверки на стороне сервера, а не в промпте: получателя и файл выбирает не
    модель. «recipient» — адрес вне списка разрешённых, «file» — файла нет в
    корпусе, он не документ или больше лимита Telegram.
    """
    call = _find_dangerous_call(message)
    if call is None:
        return None
    recipients = requested_recipients(call)
    if not recipients:
        return "recipient"
    allowed = permitted_chat_ids(config)
    if any(chat_id not in allowed for chat_id in recipients):
        return "recipient"
    if call["name"] == SEND_DOCUMENT_TOOL:
        file_name = str((call.get("args") or {}).get("file_name") or "").strip()
        if resolve_document is None or resolve_document(file_name) is None:
            return "file"
    return None


def build_agent(
    checkpointer: Any,
    model: BaseChatModel,
    tools: list[BaseTool],
    send_telegram_fn: SendTelegramFn,
    resolve_document: ResolveDocumentFn | None = None):
    """Компилирует персистентный ReAct-граф с HIL-гейтом на отправку наружу.

    `tools` — безопасные инструменты (get_current_time, search_knowledge_base,
    find_recipient). Опасные `send_telegram_message` / `send_telegram_document`
    исполняются не в `execute_tool`, а через ветку с `interrupt`: `route_after_model`
    проверяет получателя по списку из конфига запроса (`allowed_recipients`) и
    наличие файла в корпусе (`resolve_document`), при промахе уводит вызов в
    `reject_send`, и только затем человека спрашивают подтверждение."""
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
            if call["name"] in DANGEROUS_TOOLS:
                continue  # опасные инструменты идут через HIL-ветку, не здесь
            if call["name"] not in tool_by_name:
                content = f"error: unknown tool '{call['name']}'"
            else:
                content = str(await tool_by_name[call["name"]].ainvoke(call["args"]))
            messages.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
            results.append(
                {"name": call["name"], "args": call["args"], "result": content}
            )
        return {"messages": messages, "tool_results": results}

    async def prepare_send(
        state: PersistentAgentState, config: RunnableConfig
    ) -> dict:
        """Idempotent: собирает payload отправки из tool_call. Без side-effect.

        Получатели и файл уже проверены в `route_after_model`; здесь к ним
        добавляются имена из списка пользователей — человек видит их в превью и
        по ним подтверждает, что адресаты те (и что их столько, сколько нужно).
        """
        call = _find_dangerous_call(state["messages"][-1])
        if call is None:
            raise ValueError("в сообщении нет опасного tool_call")
        args = call["args"]
        titles = allowed_recipients(config)
        chat_ids = requested_recipients(call)
        draft: dict = {
            "chat_ids": chat_ids,
            # Готовый список для превью и результата: имя + chat_id по каждому.
            "recipients": [
                {"chat_id": chat_id, "title": titles.get(chat_id, "")}
                for chat_id in chat_ids
            ],
            "tool_call_id": call["id"],
        }
        if call["name"] == SEND_DOCUMENT_TOOL:
            file_name = str(args.get("file_name", "")).strip()
            resolved = resolve_document(file_name) if resolve_document else None
            draft.update(
                {
                    "kind": "document",
                    "file_name": file_name,
                    "path": str(resolved) if resolved else "",
                    "caption": args.get("caption", ""),
                }
            )
        else:
            draft.update({"kind": "message", "text": args.get("text", "")})
        return {"draft": draft}

    async def reject_send(state: PersistentAgentState, config: RunnableConfig) -> dict:
        """Вызов не прошёл проверку — отправки нет, агенту уходит ошибка.

        Узел возвращает ToolMessage, чтобы модель увидела отказ инструмента и
        исправилась (цикл ReAct): нашла человека через `find_recipient`, взяла
        корректное имя файла из `search_knowledge_base` — вместо того чтобы
        «додумывать» адресата или документ.
        """
        last = state["messages"][-1]
        call = _find_dangerous_call(last)
        reason = rejection_reason(last, config, resolve_document)
        requested = requested_recipient(last)
        recipients = allowed_recipients(config)
        initiator = str(_configurable(config).get("delivery_chat_id") or "").strip()
        tool_name = (call or {}).get("name", DANGEROUS_TOOL)

        if reason == "file":
            file_name = str((call or {}).get("args", {}).get("file_name") or "—")
            content = (
                f"отправка отклонена: файл {file_name!r} не найден в базе знаний "
                "(или это не документ / он больше лимита Telegram). Возьми точное "
                "имя из поля sources[].file_name результата search_knowledge_base."
            )
        else:
            named = sorted(
                f"{chat_id} ({title})" if title else chat_id
                for chat_id, title in recipients.items()
            )
            # Модели показываем имена, а не только цифры: чат-инициатор обычно
            # подписан «этот чат», список людей — их именами из адресной книги.
            available = "; ".join(named) or "нет"
            content = (
                f"отправка отклонена: получатель {requested or '—'} не в списке "
                f"разрешённых. Доступные получатели: {available}."
            )
            if initiator:
                content += f" Повтори вызов с chat_id={initiator}."
        logger.warning(
            "Отклонена отправка (%s): запрошен chat_id=%s, разрешены %s",
            reason or "—",
            requested or "—",
            ", ".join(sorted(recipients)) or "—",
        )
        return {
            "messages": [
                ToolMessage(content=content, tool_call_id=(call or {}).get("id", ""), name=tool_name)
            ],
            "tool_results": [
                {
                    "name": tool_name,
                    "args": {"chat_id": requested},
                    "result": content,
                }
            ],
        }

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
        sent = False
        if approved:
            try:
                outcome = await send_telegram_fn(draft)  # SIDE-EFFECT только здесь
                sent = True
                content = (
                    outcome
                    if isinstance(outcome, str) and outcome
                    else "отправлено"
                )
            except Exception as exc:  # доставка не удалась — не выдаём успех
                logger.warning("Отправка не удалась: %s", exc)
                content = f"не удалось отправить: {exc}"
        else:
            content = "отправка отменена пользователем"
        # Имя инструмента — по виду отправки: в отчёте агента и в логе должно быть
        # видно, ушёл файл или текст.
        tool_name = (
            SEND_DOCUMENT_TOOL if draft.get("kind") == "document" else SEND_MESSAGE_TOOL
        )
        return {
            "sent": sent,
            "messages": [
                ToolMessage(content=content, tool_call_id=draft.get("tool_call_id", ""))
            ],
            "tool_results": [
                {"name": tool_name, "args": draft, "result": content}
            ],
        }
	
    async def force_finish(state: PersistentAgentState) -> dict:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return {"messages": [AIMessage(content="Превышен лимит итераций")]}
        return {}

    def route_after_model(
        state: PersistentAgentState, config: RunnableConfig
    ) -> Literal["execute_tool", "prepare_send", "reject_send", "force_finish"]:
        if state["iteration_count"] >= MAX_ITERATIONS:
            return "force_finish"
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None)
        if not calls:
            return "force_finish"
        if any(call["name"] in DANGEROUS_TOOLS for call in calls):
            # И получателя, и (для документа) файл проверяет сервер, а не модель.
            if rejection_reason(last, config, resolve_document) is None:
                return "prepare_send"
            return "reject_send"
        return "execute_tool"

    builder = StateGraph(PersistentAgentState)
    builder.add_node("call_model", call_model)
    builder.add_node("execute_tool", execute_tool)
    builder.add_node("prepare_send", prepare_send)
    builder.add_node("reject_send", reject_send)
    builder.add_node("confirm_and_send", confirm_and_send)
    builder.add_node("force_finish", force_finish)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        route_after_model,
        {
            "execute_tool": "execute_tool",
            "prepare_send": "prepare_send",
            "reject_send": "reject_send",
            "force_finish": "force_finish",
        },
    )
    builder.add_edge("execute_tool", "call_model")
    builder.add_edge("reject_send", "call_model")  # отказ возвращается модели
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
    resolve_document: ResolveDocumentFn | None = None,
    sqlite_path: str = "var/agent_checkpoints.sqlite",
    postgres_url: str = "",
) -> AsyncIterator[Any]:
    """Поднимает нужный чекпоинтер и отдаёт скомпилированный граф.
    `setup()` вызывается ровно один раз здесь — не на каждый запрос.
    """
    if backend == "memory":
        # InMemorySaver не требует setup() и живёт в памяти процесса.
        yield build_agent(
            InMemorySaver(), model, tools, send_telegram_fn,
            resolve_document=resolve_document,
        )
    elif backend == "sqlite":
        from pathlib import Path

        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(sqlite_path) as saver:
            await saver.setup()
            yield build_agent(
                saver, model, tools, send_telegram_fn,
                resolve_document=resolve_document,
            )
    elif backend == "postgres":
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        async with AsyncPostgresSaver.from_conn_string(
            postgres_url
        ) as saver:
            await saver.setup()
            yield build_agent(
                saver, model, tools, send_telegram_fn,
                resolve_document=resolve_document,
            )
    else:
        raise ValueError(f"неизвестный AGENT_CHECKPOINTER: {backend!r}")