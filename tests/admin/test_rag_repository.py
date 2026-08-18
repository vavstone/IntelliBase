"""Тесты RAG-аналитики AdminRepository (Б5.5): log_rag_query, knowledge_gaps."""

from unittest.mock import MagicMock

import pytest

from app.admin.repository import AdminRepository
from app.chat.repositories.pg_models import RagQueryRow


class _FakeSession:
    def __init__(self, execute_result=None) -> None:
        self.added: list = []
        self._execute_result = execute_result

    def add(self, obj) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        pass

    async def execute(self, stmt):
        return self._execute_result

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a) -> None:
        pass


class _FakeSessionFactory:
    def __init__(self, execute_result=None) -> None:
        self._execute_result = execute_result
        self.last_session: _FakeSession | None = None

    def __call__(self):
        self.last_session = _FakeSession(self._execute_result)
        return self.last_session


@pytest.mark.asyncio
async def test_log_rag_query_normalizes_question() -> None:
    sf = _FakeSessionFactory()
    repo = AdminRepository(sf)
    await repo.log_rag_query("  Что Такое ТРОИС?  ", True, 0.86)

    assert sf.last_session is not None
    row = sf.last_session.added[0]
    assert isinstance(row, RagQueryRow)
    assert row.question_normalized == "что такое троис?"
    assert row.confident is True
    assert row.top_score == 0.86


@pytest.mark.asyncio
async def test_log_rag_query_noop_without_session_factory() -> None:
    repo = AdminRepository(None)
    await repo.log_rag_query("x", True, 0.5)  # не падает


@pytest.mark.asyncio
async def test_knowledge_gaps_returns_list() -> None:
    result = MagicMock()
    result.scalars.return_value.all.return_value = ["вопрос1", "вопрос2"]
    sf = _FakeSessionFactory(execute_result=result)
    repo = AdminRepository(sf)
    gaps = await repo.knowledge_gaps(limit=10)
    assert gaps == ["вопрос1", "вопрос2"]


@pytest.mark.asyncio
async def test_knowledge_gaps_empty_without_session_factory() -> None:
    repo = AdminRepository(None)
    assert await repo.knowledge_gaps() == []
