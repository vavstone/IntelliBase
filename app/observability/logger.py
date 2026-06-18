import logging
import structlog

def setup_logging(level: str = "INFO") -> None:
    # Базовый уровень для стандартного логирования
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))

    # Единая цепочка процессоров
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.JSONRenderer(ensure_ascii=False),
    ]

    # Настройка structlog для логгеров, создаваемых через structlog.get_logger()
    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Настройка стандартных логгеров (httpx, uvicorn, root и др.)
    # foreign_pre_chain – процессоры, применяемые до финального рендерера
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=processors[-1],          # JSONRenderer
        foreign_pre_chain=processors[:-1], # остальные процессоры (без JSONRenderer)
    )

    # Применяем форматтер к корневому логгеру
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))