import logging.config
from pythonjsonlogger.json import JsonFormatter
from .config import get_settings

class CustomJsonFormatter(JsonFormatter):
    def formatTime(self, record, datefmt=None):
        # Получаем обычную строку времени без миллисекунд через родительский метод
        # (стандартный logging.Formatter.formatTime с заданным datefmt)
        base_time = super().formatTime(record, datefmt)
        # Добавляем миллисекунды
        msecs = record.msecs
        return f"{base_time}:{int(msecs):03d}"


def setup_logging():
    settings = get_settings()
    log_path = settings.log_path
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / "app.log"

    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": CustomJsonFormatter,
                "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
                "json_ensure_ascii": False,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "json_file": {
                "class": "logging.FileHandler",
                "filename": str(log_file),
                "mode": "a",
                "formatter": "json",
                "encoding": "utf-8",
            }
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["json_file"],
        },
    }
    logging.config.dictConfig(LOGGING_CONFIG)