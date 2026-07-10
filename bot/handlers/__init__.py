"""
Регистрация роутеров в Dispatcher.
"""

from aiogram import Dispatcher

from . import commands, fsm, text

__all__ = ["register_routers"]


def register_routers(dp: Dispatcher) -> None:
    dp.include_router(commands.router)
    dp.include_router(fsm.router)
    dp.include_router(text.router)
