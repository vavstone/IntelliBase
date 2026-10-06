import pytest
import pytest_asyncio
from unittest.mock import AsyncMock
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.services.agent_persistent import (
    allowed_recipients,
    build_agent,
    permitted_chat_ids,
    requested_recipient,
)
from app.tools.graph_tools import get_current_time

INITIATOR = "100000001"  # чат, из которого пришёл запрос (идентификаторы синтетические)
IVANOV = "111222333"  # получатель из списка пользователей бота (адресная книга)
FOREIGN = "999000111"  # чужой чат: ни в списке, ни инициатор


class FakeChat:
    """Content-aware заглушка: пока нет результата инструмента — просит отправку,
    после — финализирует. Устойчива к перезапуску узла при resume.

    Аргументы вызова задаются тестом: так проверяется, что получателя и файл
    выбирает сервер, а не модель."""

    def __init__(
        self,
        chat_ids: list[str] | None = None,
        *,
        tool: str = "send_telegram_message",
        file_name: str = "",
    ) -> None:
        self.chat_ids = chat_ids if chat_ids is not None else [FOREIGN]
        self.tool = tool
        self.file_name = file_name
        self.calls = 0

    def bind_tools(self, tools):  # noqa: ANN001
        return self

    async def ainvoke(self, messages):  # noqa: ANN001
        self.calls += 1
        if any(getattr(m, "type", "") == "tool" for m in messages):
            return AIMessage(content="Готово, сообщение обработано.", id="ai-final")
        if self.tool == "send_telegram_document":
            args: dict = {"chat_ids": self.chat_ids, "file_name": self.file_name}
        else:
            args = {"chat_ids": self.chat_ids, "text": "Вот информация из базы знаний..."}
        return AIMessage(
            content="",
            id="ai-send",
            tool_calls=[
                {
                    "name": self.tool,
                    "args": args,
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        )


DOCUMENT_IN_CORPUS = "ПС Тарифы. Технические требования. Версия 2.4.docx"


def _resolve_document(name: str):
    """Заглушка корпуса: существует ровно один документ (проверяем политику, не ФС)."""
    return f"/corpus/{name}" if name == DOCUMENT_IN_CORPUS else None


async def _graph(chat_ids: list[str] | str, send_result: str | None = None, **chat_kwargs):
    """Граф со sqlite-чекпоинтером в памяти + мок отправки."""
    if isinstance(chat_ids, str):
        chat_ids = [chat_ids]
    saver_cm = AsyncSqliteSaver.from_conn_string(":memory:")
    saver = await saver_cm.__aenter__()
    await saver.setup()
    send_fn = AsyncMock(return_value=send_result or "отправлено")
    g = build_agent(
        saver,
        FakeChat(chat_ids, **chat_kwargs),
        [get_current_time],
        send_fn,
        resolve_document=_resolve_document,
    )
    return g, send_fn, saver_cm


@pytest_asyncio.fixture
async def graph_initiator():
    """Модель просит отправку в чат инициатора — разрешённый получатель."""
    g, send_fn, cm = await _graph(INITIATOR)
    yield g, send_fn
    await cm.__aexit__(None, None, None)


@pytest_asyncio.fixture
async def graph_foreign():
    """Модель просит отправку на чужой chat_id (нет в списке)."""
    g, send_fn, cm = await _graph(FOREIGN)
    yield g, send_fn
    await cm.__aexit__(None, None, None)


def _initial() -> dict:
    return {"messages": [HumanMessage("отправь ...")], "iteration_count": 0,
            "tool_results": [], "draft": None, "sent": False}


def _config(thread_id: str, initiator: str | None = INITIATOR, recipients=None) -> dict:
    """Конфиг как его собирает роут: получатели из БД + инициатор запроса."""
    return {
        "configurable": {
            "thread_id": thread_id,
            "delivery_chat_id": initiator,
            "allowed_recipients": (
                {IVANOV: "Иванов Пётр, руководитель"} if recipients is None else recipients
            ),
        }
    }


@pytest.mark.asyncio
async def test_reaches_interrupt(graph_initiator):
    g, _ = graph_initiator
    result = await g.ainvoke(_initial(), _config("t1"))
    assert "__interrupt__" in result          # payload ушёл наружу
    snap = await g.aget_state(_config("t1"))
    assert snap.next == ("confirm_and_send",)  # ждёт человека


@pytest.mark.asyncio
async def test_resume_true_sends(graph_initiator):
    g, send_fn = graph_initiator
    config = _config("t2")
    await g.ainvoke(_initial(), config)                     # дошли до interrupt
    result = await g.ainvoke(Command(resume=True), config)  # одобрили
    assert result["sent"] is True
    send_fn.assert_called_once()          # side-effect выполнен ровно один раз


@pytest.mark.asyncio
async def test_resume_false_skips_side_effect(graph_initiator):
    g, send_fn = graph_initiator
    config = _config("t3")
    await g.ainvoke(_initial(), config)
    result = await g.ainvoke(Command(resume=False), config)
    assert result["sent"] is False
    send_fn.assert_not_called()           # ← главное требование задания


# --- политика получателя: адресата выбирает сервер, не LLM ---


@pytest.mark.asyncio
async def test_foreign_recipient_is_rejected_without_interrupt(graph_foreign):
    """Модель попросила отправить на чужой chat_id — отправки нет вообще.

    Ни паузы (подтверждать нечего), ни side-effect: агенту возвращается ошибка
    инструмента, чтобы он нашёл получателя через find_recipient или переспросил.
    """
    g, send_fn = graph_foreign
    result = await g.ainvoke(_initial(), _config("t4"))

    assert "__interrupt__" not in result
    assert result["sent"] is False
    send_fn.assert_not_called()
    errors = [t for t in result["tool_results"] if "отклонена" in t["result"]]
    assert errors, result["tool_results"]
    # Отказ называет и запрещённый адрес, и доступных получателей — с именами.
    assert FOREIGN in errors[0]["result"]
    assert INITIATOR in errors[0]["result"]
    assert "Иванов Пётр" in errors[0]["result"]


@pytest.mark.asyncio
async def test_receipient_from_address_book_is_allowed():
    """Получатель из списка пользователей бота проходит, даже если это не инициатор."""
    g, send_fn, cm = await _graph(IVANOV, send_result="сообщение отправлено в чат 111222333")
    try:
        config = _config("t5")
        result = await g.ainvoke(_initial(), config)
        assert "__interrupt__" in result  # пауза: получатель разрешён
        await g.ainvoke(Command(resume=True), config)
        send_fn.assert_called_once()
    finally:
        await cm.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_preview_contains_recipient_title():
    """В превью уходят имена получателей: человек подтверждает по ним, не по цифрам."""
    g, _, cm = await _graph(IVANOV, send_result="ok")
    try:
        result = await g.ainvoke(_initial(), _config("t6"))
        preview = result["__interrupt__"][0].value["preview"]
        assert preview["recipients"] == [
            {"chat_id": IVANOV, "title": "Иванов Пётр, руководитель"}
        ]
    finally:
        await cm.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_several_recipients_in_one_send():
    """«Отправь мне и Иванову» — одна отправка, одна пауза, оба получателя."""
    g, send_fn, cm = await _graph([INITIATOR, IVANOV], send_result="отправлено")
    try:
        config = _config("t10")
        result = await g.ainvoke(_initial(), config)

        preview = result["__interrupt__"][0].value["preview"]
        assert [r["chat_id"] for r in preview["recipients"]] == [INITIATOR, IVANOV]
        await g.ainvoke(Command(resume=True), config)
        send_fn.assert_called_once()  # один side-effect на всю рассылку
        assert send_fn.call_args.args[0]["chat_ids"] == [INITIATOR, IVANOV]
    finally:
        await cm.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_one_foreign_recipient_rejects_whole_send():
    """Если хоть один адресат не разрешён — не отправляем никому (а не «частично»)."""
    g, send_fn, cm = await _graph([INITIATOR, FOREIGN])
    try:
        result = await g.ainvoke(_initial(), _config("t11"))

        assert "__interrupt__" not in result
        send_fn.assert_not_called()
        assert "отклонена" in result["tool_results"][0]["result"]
    finally:
        await cm.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_document_pause_shows_file_and_recipients():
    """Отправка документа: в превью видно файл и получателя."""
    g, _, cm = await _graph(
        [IVANOV], tool="send_telegram_document", file_name=DOCUMENT_IN_CORPUS
    )
    try:
        result = await g.ainvoke(_initial(), _config("t12"))

        preview = result["__interrupt__"][0].value["preview"]
        assert preview["kind"] == "document"
        assert preview["file_name"] == DOCUMENT_IN_CORPUS
        assert preview["path"] == f"/corpus/{DOCUMENT_IN_CORPUS}"

        resumed = await g.ainvoke(Command(resume=True), _config("t12"))
        # В отчёте видно, что ушёл именно файл, а не текст.
        assert resumed["tool_results"][0]["name"] == "send_telegram_document"
    finally:
        await cm.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_document_missing_in_corpus_is_rejected():
    """Файла нет в корпусе — паузы и отправки нет, агенту уходит ошибка."""
    g, send_fn, cm = await _graph(
        [IVANOV], tool="send_telegram_document", file_name="ФТ на Тарифы-1.docx"
    )
    try:
        result = await g.ainvoke(_initial(), _config("t13"))

        assert "__interrupt__" not in result
        send_fn.assert_not_called()
        assert "не найден в базе знаний" in result["tool_results"][0]["result"]
    finally:
        await cm.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_no_initiator_and_empty_list_rejects(graph_foreign):
    """Пустой список и нет chat_id в запросе — отправлять некому."""
    g, send_fn = graph_foreign
    result = await g.ainvoke(_initial(), _config("t7", initiator=None, recipients={}))

    assert "__interrupt__" not in result
    send_fn.assert_not_called()


@pytest.mark.asyncio
async def test_initiator_allowed_even_without_db_list():
    """Инициатор разрешён всегда: «отправь мне» не зависит от доступности БД."""
    g, send_fn, cm = await _graph(INITIATOR)
    try:
        config = _config("t8", recipients={})  # БД пуста/недоступна
        result = await g.ainvoke(_initial(), config)
        assert "__interrupt__" in result
        await g.ainvoke(Command(resume=True), config)
        send_fn.assert_called_once()
    finally:
        await cm.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_approve_with_failed_delivery_is_not_reported_as_sent():
    """Ошибка доставки → sent=False и честный текст, а не «отправлено»."""
    g, send_fn, cm = await _graph(INITIATOR)
    send_fn.side_effect = RuntimeError("бот недоступен")
    try:
        config = _config("t9")
        await g.ainvoke(_initial(), config)
        result = await g.ainvoke(Command(resume=True), config)

        assert result["sent"] is False
        assert "не удалось отправить" in result["tool_results"][-1]["result"]
    finally:
        await cm.__aexit__(None, None, None)


def test_allowed_recipients_merges_list_and_initiator() -> None:
    config = {
        "configurable": {
            "delivery_chat_id": INITIATOR,
            "allowed_recipients": {IVANOV: "Иванов Пётр"},
        }
    }
    assert allowed_recipients(config) == {IVANOV: "Иванов Пётр", INITIATOR: ""}
    assert permitted_chat_ids(config) == {IVANOV, INITIATOR}
    # Без инициатора остаётся только список из БД.
    assert permitted_chat_ids({"configurable": {"allowed_recipients": {IVANOV: "И"}}}) == {IVANOV}
    assert permitted_chat_ids(None) == set()


def test_requested_recipient_reads_tool_call() -> None:
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "send_telegram_message",
                "args": {"chat_id": " 42 ", "text": "x"},
                "id": "c1",
                "type": "tool_call",
            }
        ],
    )
    assert requested_recipient(message) == "42"
