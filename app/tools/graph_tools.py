"""Инструменты LangGraph агента.

Здесь живут три инструмента, список `TOOLS`.
"""
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import functools

from langchain_core.tools import tool

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
def send_telegram_message(chat_id: str, text: str) -> str:
    """Отправляет текстовое сообщение клиенту в Telegram по идентификатору чата.
    Вызывай только для финального ответа клиенту и только после того, как все нужные данные уже собраны.
    Возвращает строку вида 'Сообщение отправлено в {chat_id}'.
    Не отправляет уведомления по другим каналам (кроме чата телеграм), не изменяет данные."""
    print(f"[TELEGRAM → {chat_id}] {text}")
    return f"Сообщение отправлено в {chat_id}"

# Список инструментов для агента.
TOOLS = [search_knowledge_base, get_current_time, send_telegram_message]