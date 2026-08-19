"""Тесты каталога категорий (ПС): normalize_slug + роуты /categories + репозиторий.

Без живой Postgres: session_factory подменяется фейком, возвращающим
фиксированные строки и записывающим SQL-вызовы. Роуты тестируются на
минимальном FastAPI-приложении (без lifespan/middleware), чтобы не мешать
прочие зависимости (observability-мидлварь тоже ходит в session_factory).
"""

from fastapi import FastAPI
import pytest
from httpx import ASGITransport, AsyncClient

from app.kb.domain import KbCategory, normalize_slug
from app.kb.repository import KbCategoryRepository


# ── normalize_slug ───────────────────────────────────────────────────────

def test_normalize_slug_lowercases_and_keeps_latin() -> None:
    assert normalize_slug("Tarify") == "tarify"
    assert normalize_slug("tamozhnya_pravo") == "tamozhnya_pravo"


def test_normalize_slug_strips_traversal() -> None:
    # '.' и '/' вне разрешённого алфавита — обхода каталога не происходит.
    assert normalize_slug("../../evil") == "evil"


def test_normalize_slug_falls_back_to_raznoe() -> None:
    assert normalize_slug(None) == "raznoe"
    assert normalize_slug("") == "raznoe"
    assert normalize_slug("Малахит") == "raznoe"  # не-латиница -> fallback
    assert normalize_slug("  Таможня и Право  ") == "raznoe"


# ── фейк сессии (без Postgres) ───────────────────────────────────────────

class _Row:
    def __init__(self, slug: str, title: str) -> None:
        self.slug = slug
        self.title = title


class _Result:
    def __init__(self, rows: list[_Row]) -> None:
        self._rows = rows

    def scalars(self) -> "_Result":
        return self

    def all(self) -> list[_Row]:
        return self._rows

    def scalar_one(self) -> _Row:
        return self._rows[0]


class _RecordingSession:
    def __init__(self, rows: list[_Row]) -> None:
        self._rows = rows
        self.executed: list = []

    async def __aenter__(self) -> "_RecordingSession":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def execute(self, stmt, params=None) -> _Result:
        self.executed.append((stmt, params))
        return _Result(self._rows)

    async def commit(self) -> None:
        pass


class _Factory:
    def __init__(self, rows: list[_Row]) -> None:
        self._rows = rows
        self.last_session: _RecordingSession | None = None

    def __call__(self) -> _RecordingSession:
        self.last_session = _RecordingSession(self._rows)
        return self.last_session


# ── репозиторий ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_categories_maps_rows() -> None:
    repo = KbCategoryRepository(
        _Factory([_Row("tarify", "Тарифы"), _Row("malahit", "Малахит")])
    )
    assert await repo.list_categories() == [
        KbCategory(slug="tarify", title="Тарифы"),
        KbCategory(slug="malahit", title="Малахит"),
    ]


@pytest.mark.asyncio
async def test_list_categories_empty_without_session_factory() -> None:
    repo = KbCategoryRepository(None)
    assert await repo.list_categories() == []


@pytest.mark.asyncio
async def test_upsert_category_uses_on_conflict_do_nothing() -> None:
    factory = _Factory([_Row("raznoe", "Разное")])
    repo = KbCategoryRepository(factory)
    result = await repo.upsert_category("raznoe", "Разное")
    assert result == KbCategory(slug="raznoe", title="Разное")
    # два вызова: INSERT (идемпотентный) + SELECT возвращаемой записи
    assert factory.last_session is not None
    assert len(factory.last_session.executed) == 2
    assert "ON CONFLICT (slug) DO NOTHING" in str(factory.last_session.executed[0][0])


# ── роуты ────────────────────────────────────────────────────────────────

def _build_app(factory: _Factory | None) -> FastAPI:
    from app.routers.categories import router

    app = FastAPI()
    app.include_router(router)
    app.state.session_factory = factory
    return app


@pytest.mark.asyncio
async def test_list_categories_route() -> None:
    factory = _Factory([_Row("tarify", "Тарифы"), _Row("malahit", "Малахит")])
    app = _build_app(factory)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.get("/categories")
    assert resp.status_code == 200
    assert resp.json() == [
        {"slug": "tarify", "title": "Тарифы"},
        {"slug": "malahit", "title": "Малахит"},
    ]


@pytest.mark.asyncio
async def test_list_categories_route_503_without_pg() -> None:
    app = _build_app(None)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.get("/categories")
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_create_category_route_normalizes_slug() -> None:
    factory = _Factory([_Row("raznoe", "Разное")])
    app = _build_app(factory)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.post(
            "/categories", json={"slug": "PostKontrol", "title": "Постконтроль"}
        )
    assert resp.status_code == 201
    # slug канонизирован (upper -> lower) ДО вызова upsert_category.
    insert = factory.last_session.executed[0]
    assert insert[1]["slug"] == "postkontrol"
    assert insert[1]["title"] == "Постконтроль"
