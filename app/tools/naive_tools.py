"""Инструменты наивного агента: реализации + описания в JSON Schema.

Здесь живут три инструмента, словарь-allowlist `DISPATCH` и список `TOOLS`
с описаниями в формате OpenAI Chat Completions. Агент (`agent_naive.run_agent`)
импортирует `TOOLS` и `DISPATCH` отсюда и сам не знает деталей реализации.
"""

from datetime import datetime
from zoneinfo import ZoneInfo
import asyncio


from app.services.rag import RAGService
from app.core.config import get_settings

settings = get_settings()
rag_service = RAGService(settings)
rag_service.build()


def search_knowledge_base(query: str) -> str:
    """Поиск ответа во внутренней базе знаний по ключевому слову запроса."""
    normalized = query.lower()
    return asyncio.run(rag_service.answer(normalized))


def get_current_time(timezone: str = "Europe/Moscow") -> str:
    """Текущие дата и время в указанном часовом поясе в формате ISO 8601."""
    now = datetime.now(ZoneInfo(timezone))
    return now.isoformat()


def send_telegram_message(chat_id: str, text: str) -> str:
    """Отправка сообщения клиенту в Telegram (в этом задании — заглушка)."""
    print(f"[TELEGRAM → {chat_id}] {text}")
    return f"Сообщение отправлено в {chat_id}"


# Allowlist: имя инструмента -> реализация. Никаких eval/getattr —
# модель может вызвать только то, что явно перечислено здесь.
DISPATCH = {
    "search_knowledge_base": search_knowledge_base,
    "get_current_time": get_current_time,
    "send_telegram_message": send_telegram_message,
}

# Описания инструментов для Chat Completions. От качества description
# напрямую зависит, выберет ли модель нужный инструмент в нужный момент.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Ищет ответ во внутренней базе знаний компании: документы, "
                "приказы, спецификации, описания. Вызывай, когда нужны "
                "фактические данные о предметной области ФТС."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос на русском языке",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "Возвращает текущие дату и время в указанном часовом поясе в формате "
                "ISO 8601. Вызывай, когда нужно знать текущее время, например для "
                "расчёта сроков возврата или доставки."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Имя часового пояса IANA, например Europe/Moscow",
                        "default": "Europe/Moscow",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_telegram_message",
            "description": (
                "Отправляет текстовое сообщение клиенту в Telegram по идентификатору "
                "чата. Вызывай только для финального ответа клиенту и только после "
                "того, как все нужные данные уже собраны."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chat_id": {
                        "type": "string",
                        "description": "Идентификатор чата клиента в Telegram",
                    },
                    "text": {
                        "type": "string",
                        "description": "Текст сообщения для клиента",
                    },
                },
                "required": ["chat_id", "text"],
            },
        },
    },
]