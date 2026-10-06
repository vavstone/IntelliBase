"""Инструменты LangGraph агента.

Здесь живут три инструмента, список `TOOLS`, и фабрика `find_recipient`
(ей нужна сессия БД, поэтому создаётся в lifespan, а не на импорте).
"""
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import functools

from langchain_core.tools import BaseTool, tool

from app.services.bot_users import search_users
from app.services.corpus_files import list_corpus_documents
from app.services.rag import RAGService
from app.core.config import get_settings

@functools.lru_cache(maxsize=1)
def _get_rag() -> RAGService:
    """Лениво создаёт и кэширует экземпляр RAGService."""
    settings = get_settings()
    rag = RAGService(settings)
    rag.build()
    return rag


@tool
async def search_knowledge_base(query: str) -> str:
    """Ищет ответ во внутренней базе знаний компании: документы, приказы, спецификации, описания.
    Вызывай, когда нужны фактические данные о предметной области ФТС.
    Возвращает словарь {answer, top_score, sources[id,file_name,page,score,snippet], confident}.
    Не отправляет уведомления, не изменяет данные."""
    rag = _get_rag()
    result = await rag.answer(query.lower())
    return json.dumps(result, ensure_ascii=False)


@tool
def get_current_time(timezone: str = "Europe/Moscow") -> str:
    """Рассчитывает текущие дату и время.
    Вызывай, когда нужно знать текущее время.
    Возвращает текущие дату и время в указанном часовом поясе в формате ISO 8601.
    Не отправляет уведомления, не изменяет данные."""
    now = datetime.now(ZoneInfo(timezone))
    return now.isoformat()


@tool
def send_telegram_message(chat_ids: list[str], text: str) -> str:
    """Отправляет текстовое сообщение в Telegram одному или нескольким получателям.
    `chat_ids` — список идентификаторов чатов: один элемент, если получатель один,
    несколько, если просят отправить нескольким («мне и Иванову»). Идентификаторы
    берутся из инструмента find_recipient (или из служебной подсказки о чате-инициаторе) —
    выдумывать их нельзя.
    Вызывай только для финального ответа и только когда все нужные данные собраны.
    Не отправляет уведомления по другим каналам, не изменяет данные."""
    joined = ", ".join(chat_ids)
    print(f"[TELEGRAM → {joined}] {text}")
    return f"Сообщение отправлено: {joined}"


@tool
def send_telegram_document(chat_ids: list[str], file_name: str, caption: str = "") -> str:
    """Отправляет файл из базы знаний в Telegram вложением — одному или нескольким получателям.
    `file_name` — имя документа ровно так, как оно пришло в поле sources[].file_name
    результата search_knowledge_base или в list_documents («ПС Тарифы. Технические
    требования. Версия 2.4.docx»); путь указывать нельзя — файл ищет сервер в корпусе.
    `chat_ids` — список чатов (несколько, если отправить нужно нескольким людям).
    `caption` — короткая подпись: что это за документ и чем полезен.
    Вызывай, когда просят «отправить документ/файл», а не выдержку текстом.
    Не изменяет документы, не отправляет ничего за пределы разрешённых получателей."""
    joined = ", ".join(chat_ids)
    print(f"[TELEGRAM DOC → {joined}] {file_name}")
    return f"Документ {file_name} отправлен: {joined}"

def make_list_documents_tool(settings) -> BaseTool:
    """Инструмент «какие документы есть в базе знаний».

    Отвечает детерминированно — по файлам корпуса, а не по выборке чанков из
    векторного поиска: на вопрос «какие ФТ у нас есть» RAG отдаёт top-K кусков
    и полного списка не даёт, модель достраивала бы его догадками. Заодно
    возвращает точные имена файлов — их и требует `send_telegram_document`.
    """

    @tool
    async def list_documents(query: str = "") -> str:
        """Показывает список документов базы знаний (при необходимости — по фильтру).
        Вызывай, когда спрашивают «какие документы/ФТ/регламенты есть в базе» или
        когда нужно уточнить точное имя файла перед отправкой.
        `query` — подстрока для фильтра по имени документа или категории
        (например «тарифы», «ЦРСВЭД»); пусто — весь список.
        Возвращает JSON-список [{category, file_name, size_kb}].
        Не отправляет сообщения, не изменяет данные."""
        found = list_corpus_documents(settings.rag_data_dir, query)
        return json.dumps(found, ensure_ascii=False)

    return list_documents


def make_find_recipient_tool(session_factory) -> BaseTool:
    """Инструмент «найди получателя по имени» из списка пользователей бота.

    Нужен, чтобы «отправь Иванову отчёт» работало: модель получает chat_id по
    имени, а не выдумывает его. Ищет только среди разрешённых (активных
    пользователей) — показать модели чужой адрес инструмент не может.
    """

    @tool
    async def find_recipient(name: str) -> str:
        """Ищет получателя отчёта по имени или должности («Иванов», «руководитель»).
        Вызывай, когда нужно отправить сообщение конкретному человеку и его chat_id
        неизвестен. Возвращает JSON-список [{chat_id, title}] — найденных людей.
        Пустой список означает, что такого получателя нет или отправка ему не разрешена.
        Не отправляет уведомления, не изменяет данные."""
        found = await search_users(session_factory, name)
        return json.dumps(found, ensure_ascii=False)

    return find_recipient


# Список инструментов для агента. `find_recipient` добавляется в lifespan
# (ему нужна сессия БД) — см. app/main.py. Опасные инструменты
# (`send_telegram_message`, `send_telegram_document`) граф не исполняет: он ведёт
# их через проверку получателя и подтверждение человеком (см. agent_persistent.py).
TOOLS = [
    search_knowledge_base,
    get_current_time,
    send_telegram_message,
    send_telegram_document,
]