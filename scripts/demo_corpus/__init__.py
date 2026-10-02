"""Демонстрационный корпус: синтетические документы для базы знаний.

Корпус полностью вымышленный (организации, номера, даты, показатели) и
предназначен для запуска системы «из коробки»: после клонирования репозитория
и первого старта коллекция наполняется без доступа к реальным документам.

Состав: 18 документов в 7 категориях (ПС), форматы PDF и DOCX. Содержимое
описано декларативно в `content_*.py`, рендеринг — в `blocks.py`.

Запуск: `uv run python scripts/generate_demo_corpus.py`.
"""

from __future__ import annotations

from .blocks import Doc, find_font_dir, render
from .content_crsved import DOCS as _CRSVED
from .content_malahit import DOCS as _MALAHIT
from .content_postkontrol import DOCS as _POSTKONTROL
from .content_pravo import DOCS as _PRAVO
from .content_pravoohrana import DOCS as _PRAVOOHRANA
from .content_raznoe import DOCS as _RAZNOE
from .content_tarify import DOCS as _TARIFY

ALL_DOCS: tuple[Doc, ...] = (
    _TARIFY + _MALAHIT + _POSTKONTROL + _PRAVO + _CRSVED + _PRAVOOHRANA + _RAZNOE
)

__all__ = ["ALL_DOCS", "Doc", "find_font_dir", "render"]
