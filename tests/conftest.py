# TLS-инспекция Kaspersky подменяет сертификаты: certifi (httpx/openai) их не
# признаёт, а хранилище сертификатов ОС — признаёт. Без этого интеграционные тесты
# (tests/test_token_count.py) падают с CERTIFICATE_VERIFY_FAILED. Тот же приём,
# что в experiments/*, scripts/run_eval.py и scripts/generate_testset.py.
import os

import truststore

truststore.inject_into_ssl()

# Трейсинг в тестах выключен принудительно: `app.main` регистрирует инструментеры
# Phoenix на импорте (до старта приложения), а в тестах ни экспорт спанов, ни
# подключение к Phoenix не нужны. Переменная окружения перебивает значение из .env.
os.environ["PHOENIX_ENABLED"] = "false"


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark test as integration test requiring external API")
