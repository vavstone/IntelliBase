import asyncio
import os
import pytest
import pytest_asyncio
from pathlib import Path
from uuid import uuid4
from typing import AsyncIterator, Literal
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import text

from app.chat.repositories.json_repo import JsonChatRepository
from app.chat.repositories.pg_repo import PostgresChatRepository
from app.chat.repositories.pg_models import Base
from app.chat.repository import ChatRepository
from app.core.config import get_settings

load_dotenv()

# ===== Вспомогательные функции для тестовой БД =====

def _build_test_db_url(base_url: str, test_db_name: str = "intellibase_test") -> str:
    """Всегда заменяет имя базы данных на test_db_name."""
    parsed = urlparse(base_url)
    new_path = f"/{test_db_name}"
    return urlunparse(parsed._replace(path=new_path))


async def _ensure_test_database_exists(db_url: str) -> None:
    """Создаёт тестовую БД, если её нет."""
    parsed = urlparse(db_url)
    system_db = "postgres"
    system_url = urlunparse(parsed._replace(path=f"/{system_db}"))
    engine = create_async_engine(system_url, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as conn:
            db_name = parsed.path.lstrip("/")
            result = await conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
                {"db_name": db_name},
            )
            if result.scalar() is None:
                await conn.execute(text(f"CREATE DATABASE {db_name}"))
    finally:
        await engine.dispose()


# ===== Фикстуры для PostgreSQL =====

@pytest.fixture
def postgres_test_db_url() -> str:
    """Возвращает URL тестовой БД. Приоритет: переменная окружения DATABASE_URL_FOR_TESTS,
    иначе основная БД с заменой имени на intellibase_test.
    """
    test_url = os.getenv("DATABASE_URL_FOR_TESTS")
    if test_url:
        return test_url
    settings = get_settings()
    base = settings.database_url
    return _build_test_db_url(base, "intellibase_test")


@pytest_asyncio.fixture
async def pg_engine(postgres_test_db_url: str):
    """Движок SQLAlchemy для тестовой БД, создаёт таблицы."""
    await _ensure_test_database_exists(postgres_test_db_url)
    engine = create_async_engine(postgres_test_db_url, pool_pre_ping=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


# ===== Фикстура для JSON =====

@pytest_asyncio.fixture
async def json_repo(tmp_path: Path) -> JsonChatRepository:
    """Создаёт JSON-репозиторий во временной директории."""
    base_dir = tmp_path / "chats"
    return JsonChatRepository(base_dir)


# ===== Основная параметризованная фикстура =====

@pytest_asyncio.fixture(params=["json", "postgres"])
async def chat_repository(request, json_repo: JsonChatRepository, pg_engine) -> AsyncIterator[ChatRepository]:
    """
    Возвращает репозиторий в зависимости от параметра.
    Для 'postgres' создаёт новую сессию и очищает таблицы.
    """
    if request.param == "json":
        yield json_repo
    elif request.param == "postgres":
        async_session = async_sessionmaker(pg_engine, expire_on_commit=False)
        async with async_session() as session:
            # Очищаем таблицы перед каждым тестом
            await session.execute(text("TRUNCATE TABLE chats, chat_messages RESTART IDENTITY CASCADE;"))
            await session.commit()
            yield PostgresChatRepository(session)
    else:
        raise ValueError(f"Unknown repository type: {request.param}")


# ===== Константы =====

OWNER = "test_user_123"
INTERFACE = "telegram"
PROVIDER: Literal["openai", "ollama", "openrouter"] = "ollama"
MODEL = "gemma3:1b"
SYSTEM_PROMPT = "You are a helpful assistant."


# ===== ТЕСТЫ =====

@pytest.mark.asyncio
async def test_create_and_get_chat(chat_repository: ChatRepository):
    """1. Создание чата и чтение его обратно."""
    # Создаём чат
    created = await chat_repository.create_chat(
        owner_external_id=OWNER,
        interface=INTERFACE,
        provider=PROVIDER,
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
    )
    assert created.id is not None
    assert created.owner_external_id == OWNER
    assert created.interface == INTERFACE
    assert created.provider == PROVIDER
    assert created.model == MODEL
    assert created.system_prompt == SYSTEM_PROMPT

    # Читаем его по id
    fetched = await chat_repository.get_chat(created.id)
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.owner_external_id == OWNER
    assert fetched.interface == INTERFACE
    assert fetched.provider == PROVIDER
    assert fetched.model == MODEL
    assert fetched.system_prompt == SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_get_or_create_chat_idempotent(chat_repository: ChatRepository):
    """Повторный вызов get_or_create_chat с теми же параметрами возвращает существующий чат."""
    # Первый вызов – создаёт
    chat1 = await chat_repository.get_or_create_chat(
        owner_external_id=OWNER,
        interface=INTERFACE,
        provider=PROVIDER,
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
    )
    # Второй вызов с теми же параметрами – должен вернуть тот же чат
    chat2 = await chat_repository.get_or_create_chat(
        owner_external_id=OWNER,
        interface=INTERFACE,
        provider=PROVIDER,
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
    )
    assert chat1.id == chat2.id
    assert chat2.system_prompt == SYSTEM_PROMPT  # остался исходный

@pytest.mark.asyncio
async def test_list_messages_chronological_order(chat_repository: ChatRepository):
    """append_message + list_messages возвращают сообщения в хронологическом порядке
    Проверяет хронологический порядок (от старых к новым)."""
    chat = await chat_repository.create_chat(
        owner_external_id=OWNER,
        interface=INTERFACE,
        provider=PROVIDER,
        model=MODEL,
        system_prompt=None,
    )
    from app.chat.domain import ChatMessage

    contents = ["first", "second", "third", "fourth", "fifth"]
    for content in contents:
        await asyncio.sleep(1.0)  # 1 секунда
        msg = ChatMessage(chat_id=chat.id, role="user", content=content)
        await chat_repository.append_message(chat.id, msg)

    all_messages = await chat_repository.list_messages(chat.id, limit=10)
    assert len(all_messages) == len(contents)

    # Проверяем, что содержимое соответствует порядку добавления
    assert [m.content for m in all_messages] == contents

    # Проверяем, что времена строго возрастают
    for i in range(1, len(all_messages)):
        assert all_messages[i - 1].created_at < all_messages[i].created_at

    limit = 3
    limited = await chat_repository.list_messages(chat.id, limit=limit)
    assert len(limited) == limit
    expected = contents[-limit:]
    assert [m.content for m in limited] == expected
    for i in range(1, len(limited)):
        assert limited[i - 1].created_at < limited[i].created_at


@pytest.mark.asyncio
async def test_list_messages_limit_returns_last_n(chat_repository: ChatRepository):
    """list_messages(limit=N) отдаёт последние N, а не первые."""
    chat = await chat_repository.create_chat(
        owner_external_id=OWNER,
        interface=INTERFACE,
        provider=PROVIDER,
        model=MODEL,
        system_prompt=None,
    )
    from app.chat.domain import ChatMessage

    for i in range(1, 6):
        msg = ChatMessage(chat_id=chat.id, role="user", content=f"msg{i}")
        await chat_repository.append_message(chat.id, msg)

    limit = 3
    last_n = await chat_repository.list_messages(chat.id, limit=limit)
    assert len(last_n) == limit
    expected = ["msg3", "msg4", "msg5"]
    assert [m.content for m in last_n] == expected

    all_messages = await chat_repository.list_messages(chat.id, limit=10)
    assert all_messages[0].content == "msg1"
    assert all_messages[1].content == "msg2"
    assert last_n != all_messages[:limit]


@pytest.mark.asyncio
async def test_soft_delete_then_new_messages_visible(chat_repository: ChatRepository):
    """После soft_delete_messages list_messages пуст, но новые сообщения видны."""
    chat = await chat_repository.create_chat(
        owner_external_id=OWNER,
        interface=INTERFACE,
        provider=PROVIDER,
        model=MODEL,
        system_prompt=None,
    )
    from app.chat.domain import ChatMessage

    for i in range(3):
        msg = ChatMessage(chat_id=chat.id, role="user", content=f"old{i}")
        await chat_repository.append_message(chat.id, msg)

    # Выполняем soft delete
    await chat_repository.soft_delete_messages(chat.id)

    # После удаления список пуст
    after_delete = await chat_repository.list_messages(chat.id)
    assert len(after_delete) == 0

    # Добавляем новые сообщения
    new_msg = ChatMessage(chat_id=chat.id, role="user", content="new")
    await chat_repository.append_message(chat.id, new_msg)

    # Новое сообщение должно быть видно
    new_list = await chat_repository.list_messages(chat.id)
    assert len(new_list) == 1
    assert new_list[0].content == "new"


@pytest.mark.asyncio
async def test_get_chat_unknown_returns_none(chat_repository: ChatRepository):
    """
    get_chat(unknown_uuid) возвращает None, а не падает.
    """
    unknown_id = uuid4()
    result = await chat_repository.get_chat(unknown_id)
    assert result is None


@pytest.mark.asyncio
async def test_append_and_list_message_sources_roundtrip(chat_repository: ChatRepository):
    """RAG-цитаты (sources) переживают append → list без потерь (Б5.5).

    Регрессия этой фичи — сценарий «sources не сохраняется» (например, нет
    колонки в БД): проверяем, что список источников {id,file_name,page,score,
    snippet} читается обратно 1-в-1 и в обоих хранилищах (json и postgres).
    """
    from app.chat.domain import ChatMessage

    chat = await chat_repository.create_chat(
        owner_external_id=OWNER,
        interface=INTERFACE,
        provider=PROVIDER,
        model=MODEL,
        system_prompt=None,
    )
    sources = [
        {"id": 1, "file_name": "tariffs.pdf", "page": 3, "score": 0.86, "snippet": "КПС..."},
        {"id": 2, "file_name": "trois.md", "page": None, "score": 0.81, "snippet": "ТРОИС..."},
    ]
    msg = ChatMessage(chat_id=chat.id, role="assistant", content="Ответ [1] и [2].", sources=sources)
    await chat_repository.append_message(chat.id, msg)

    listed = await chat_repository.list_messages(chat.id, limit=10)
    assert len(listed) == 1
    assert listed[0].sources == sources