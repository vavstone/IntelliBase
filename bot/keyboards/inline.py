"""
Inline-клавиатуры для сценариев бота.
"""

from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

# Темы — примеры из диплома (поиск по корпоративной базе знаний)
DEFAULT_TOPICS: list[tuple[str, str]] = [
    ("Проектная документация", "proj_doc"),
    ("Описания БД", "db_desc"),
    ("Спецификации взаимодействия", "spec_interact"),
]


def topics_kb(
    topics: list[tuple[str, str]] | None = None,
) -> InlineKeyboardMarkup:
    """Inline-кнопки выбора темы + Отмена."""
    topics = topics or DEFAULT_TOPICS
    kb = InlineKeyboardBuilder()
    for label, slug in topics:
        kb.button(text=label, callback_data=f"topic:{slug}")
    kb.button(text="Отмена", callback_data="topic:cancel")
    kb.adjust(1)
    return kb.as_markup()
