import logging
from app.logger_setup import setup_logging

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Начало работы")
    logger.info("Конец работы")


if __name__ == '__main__':
    main()