"""Notifier: backend → bot/notify.

Тонкий клиент: POST /notify с X-Internal-Token. Используется для одиночных
push'ей (handoff-уведомления оператору, тех-алерты в админ-чат).
"""

import asyncio
from pathlib import Path

import httpx


async def notify_user(
    chat_id_tg: int,
    text: str,
    bot_url: str,
    internal_token: str,
) -> None:
    async with httpx.AsyncClient(timeout=5.0) as c:
        r = await c.post(
            f"{bot_url}/notify",
            json={"chat_id": chat_id_tg, "text": text},
            headers={"X-Internal-Token": internal_token},
        )
        r.raise_for_status()


async def deliver_document_to_bot(draft: dict, bot_url: str, internal_token: str) -> str:
    """Отправляет файл корпуса получателю через бота (multipart на /notify/document).

    Файл читается на стороне app: корпус смонтирован в app-контейнер, а в
    bot-контейнер `./data` не смонтирован — поэтому байты передаются по сети,
    а не путём. Ошибка доставки, как и у текста, поднимает `RuntimeError`.
    """
    chat_id = str(draft.get("chat_id") or "").strip()
    path = Path(str(draft.get("path") or ""))
    caption = draft.get("caption") or ""
    try:
        payload = await asyncio.to_thread(path.read_bytes)
    except OSError as exc:
        raise RuntimeError(f"файл недоступен ({exc})") from exc
    try:
        async with httpx.AsyncClient(timeout=60.0) as c:
            r = await c.post(
                f"{bot_url}/notify/document",
                data={"chat_id": chat_id, "caption": caption},
                files={"file": (path.name, payload, "application/octet-stream")},
                headers={"X-Internal-Token": internal_token},
            )
            r.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"бот недоступен ({exc})") from exc
    return f"документ «{path.name}» отправлен в чат {chat_id}"


async def deliver_to_bot(draft: dict, bot_url: str, internal_token: str) -> str:
    """Доставляет черновик агента через бота и возвращает текст результата.

    Используется как side-effect агентного графа (после HIL-подтверждения).
    При недоставке поднимает `RuntimeError`: вызывающий узел обязан пометить
    отправку неуспешной — «сообщение отправлено» не должно быть неправдой.
    Допустимость получателя проверяет граф (список пользователей бота +
    чат-инициатор запроса), здесь остаётся только доставка.
    """
    chat_id = str(draft.get("chat_id") or "").strip()
    text = draft.get("text", "")
    try:
        await notify_user(int(chat_id), text, bot_url, internal_token)
    except Exception as exc:
        raise RuntimeError(f"бот недоступен ({exc})") from exc
    return f"сообщение отправлено в чат {chat_id}"
