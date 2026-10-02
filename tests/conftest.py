# TLS-инспекция Kaspersky подменяет сертификаты: certifi (httpx/openai) их не
# признаёт, а хранилище сертификатов ОС — признаёт. Без этого интеграционные тесты
# (tests/test_token_count.py) падают с CERTIFICATE_VERIFY_FAILED. Тот же приём,
# что в experiments/*, scripts/run_eval.py и scripts/generate_testset.py.
import truststore

truststore.inject_into_ssl()


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark test as integration test requiring external API")
