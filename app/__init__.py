"""FastAPI-бэкенд IntelliBase.

Первый импорт пакета настраивает тихий режим HuggingFace — до того, как его
подтянут langchain/transformers (см. app/core/hf_env.py).
"""

from app.core import hf_env  # noqa: F401
