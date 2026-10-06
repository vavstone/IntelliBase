"""Юнит-тесты роутера агента: сборка config и начального состояния (app/routers/agent.py).

Проверяется контракт с ботом: получателя выбирает сервер (config), а служебная
подсказка о чате-инициаторе добавляется здесь, а не клиентом — иначе вызовы API
напрямую (curl, будущий UI) не знают, кто спрашивает.
"""

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
