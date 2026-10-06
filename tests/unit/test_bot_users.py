"""Юнит-тесты списка разрешённых пользователей бота (app/services/bot_users.py).

БД подменяется фейковой сессией: проверяется контракт — кто получает доступ,
что происходит при недоступной БД (fail-closed по таблице, bootstrap остаётся)
и как формируется список получателей с именами.
"""

from types import SimpleNamespace

import pytest

from app.services.bot_users import (
    is_allowed,
    load_active_users,
    parse_chat_ids,
    resolve_allowed,
)


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _Session:
    def __init__(self, rows):
        self._rows = rows

    async def execute(self, *args, **kwargs):
        return _Result(self._rows)


class _SessionContext:
    def __init__(self, rows, error: Exception | None = None):
        self._rows = rows
        self._error = error

    async def __aenter__(self):
        if self._error is not None:
            raise self._error
        return _Session(self._rows)

    async def __aexit__(self, *exc):
        return False


def _factory(rows, error: Exception | None = None):
    return lambda: _SessionContext(rows, error)


def test_parse_chat_ids() -> None:
    assert parse_chat_ids("") == set()
    assert parse_chat_ids(" 1, 2 ,3 ") == {"1", "2", "3"}
    assert parse_chat_ids("73061220") == {"73061220"}


def test_resolve_allowed_adds_bootstrap_without_overriding_names() -> None:
    allowed = resolve_allowed({"5": "Иванов Пётр"}, "7, 5")
    assert allowed == {"5": "Иванов Пётр", "7": ""}  # у bootstrap имени нет


@pytest.mark.asyncio
async def test_load_active_users_returns_mapping() -> None:
    rows = [SimpleNamespace(chat_id=111, title="Иванов Пётр")]
    assert await load_active_users(_factory(rows)) == {"111": "Иванов Пётр"}


@pytest.mark.asyncio
async def test_load_active_users_fails_closed() -> None:
    """БД недоступна → пустой список, а не «разрешаем всем»."""
    factory = _factory([], error=RuntimeError("db down"))
    assert await load_active_users(factory) == {}


@pytest.mark.asyncio
async def test_is_allowed_checks_db_and_bootstrap() -> None:
    factory = _factory([SimpleNamespace(chat_id=111, title="Иванов")])

    assert await is_allowed(factory, "111") is True           # из БД
    assert await is_allowed(factory, "777", "777,888") is True  # bootstrap
    assert await is_allowed(factory, "999") is False          # нет в списке
    assert await is_allowed(factory, "") is False             # пустой id


@pytest.mark.asyncio
async def test_is_allowed_keeps_bootstrap_when_db_is_down() -> None:
    """Упавшая БД не должна отрезать аварийный доступ из настроек."""
    factory = _factory([], error=RuntimeError("db down"))
    assert await is_allowed(factory, "777", "777") is True
    assert await is_allowed(factory, "111") is False
