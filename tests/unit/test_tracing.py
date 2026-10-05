"""Юнит-тесты подключения инструментеров Phoenix (app/observability/tracing.py).

Инструменторы и `phoenix.otel.register` подменяются фейковыми модулями: тест
проверяет контракт `setup_tracing` / `instrument_fastapi_app` (какие инструментеры
подняты, что вернулось), а не сам экспорт спанов — он проверяется живьём в UI Phoenix.
"""

import sys
import types
from types import SimpleNamespace

import pytest

import app.observability.tracing as tracing_module


class _FakeInstrumentor:
    """Запоминает факт instrument() и полученный tracer_provider."""

    instances: list["_FakeInstrumentor"] = []

    def __init__(self) -> None:
        self.tracer_provider = None
        _FakeInstrumentor.instances.append(self)

    def instrument(self, tracer_provider=None) -> None:
        self.tracer_provider = tracer_provider


@pytest.fixture(autouse=True)
def _reset_instances() -> None:
    _FakeInstrumentor.instances = []


def _module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


@pytest.fixture
def fake_phoenix(monkeypatch) -> dict:
    """Подменяет phoenix.otel.register и все три инструментера."""
    sentinel = object()
    registered: dict = {}

    def fake_register(*, project_name=None, endpoint=None, **kwargs):
        registered["project_name"] = project_name
        registered["endpoint"] = endpoint
        return sentinel

    monkeypatch.setitem(sys.modules, "phoenix.otel", _module("phoenix.otel", register=fake_register))
    monkeypatch.setitem(
        sys.modules,
        "openinference.instrumentation.openai",
        _module("openinference.instrumentation.openai", OpenAIInstrumentor=_FakeInstrumentor),
    )
    monkeypatch.setitem(
        sys.modules,
        "openinference.instrumentation.langchain",
        _module("openinference.instrumentation.langchain", LangChainInstrumentor=_FakeInstrumentor),
    )
    monkeypatch.setitem(
        sys.modules,
        "openinference.instrumentation.llama_index",
        _module(
            "openinference.instrumentation.llama_index",
            LlamaIndexInstrumentor=_FakeInstrumentor,
        ),
    )
    # Шим агентов трогает реальный llama_index — в юнит-тесте он не нужен.
    monkeypatch.setattr(tracing_module, "_shim_llama_index_base_agent", lambda: None)
    monkeypatch.setattr(tracing_module, "find_spec", lambda name: object())

    return {"sentinel": sentinel, "registered": registered}


def _settings(**overrides) -> SimpleNamespace:
    base = {
        "phoenix_enabled": True,
        "phoenix_collector_endpoint": "http://phoenix:4317",
        "phoenix_excluded_urls": "/health,/ready",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_disabled_by_flag(fake_phoenix) -> None:
    assert tracing_module.setup_tracing(_settings(phoenix_enabled=False)) is None
    assert _FakeInstrumentor.instances == []
    assert fake_phoenix["registered"] == {}


def test_empty_endpoint_is_not_enabled(fake_phoenix) -> None:
    assert tracing_module.setup_tracing(_settings(phoenix_collector_endpoint="")) is None
    assert _FakeInstrumentor.instances == []


def test_all_instrumentors_are_registered(fake_phoenix) -> None:
    assert tracing_module.setup_tracing(_settings()) is fake_phoenix["sentinel"]

    assert fake_phoenix["registered"] == {
        "project_name": "diploma-fastapi",
        "endpoint": "http://phoenix:4317",
    }
    # OpenAI, LangChain, LlamaIndex — все три получили один tracer_provider.
    assert len(_FakeInstrumentor.instances) == 3
    for instrumentor in _FakeInstrumentor.instances:
        assert instrumentor.tracer_provider is fake_phoenix["sentinel"]


def test_missing_langchain_instrumentor_keeps_others(fake_phoenix, monkeypatch) -> None:
    # fake_phoenix уже подменил find_spec на «пакет установлен»: здесь
    # выключаем ровно один модуль.
    present = tracing_module.find_spec
    monkeypatch.setattr(
        tracing_module,
        "find_spec",
        lambda name: None if name.endswith("langchain") else present(name),
    )

    assert tracing_module.setup_tracing(_settings()) is fake_phoenix["sentinel"]
    # OpenAI и LlamaIndex подняты, LangChain — нет (пакет не установлен).
    assert len(_FakeInstrumentor.instances) == 2


def test_missing_llama_index_instrumentor_keeps_others(fake_phoenix, monkeypatch) -> None:
    present = tracing_module.find_spec
    monkeypatch.setattr(
        tracing_module,
        "find_spec",
        lambda name: None if name.endswith("llama_index") else present(name),
    )

    assert tracing_module.setup_tracing(_settings()) is fake_phoenix["sentinel"]
    assert len(_FakeInstrumentor.instances) == 2


# --- FastAPI-инструментер (серверный спан запроса) ---


class _FakeFastAPIInstrumentor:
    instrumented: list[dict] = []

    @classmethod
    def instrument_app(cls, app, **kwargs) -> None:
        cls.instrumented.append({"app": app, **kwargs})


@pytest.fixture
def fake_fastapi(monkeypatch) -> _FakeFastAPIInstrumentor:
    _FakeFastAPIInstrumentor.instrumented = []
    monkeypatch.setitem(
        sys.modules,
        "opentelemetry.instrumentation.fastapi",
        _module(
            "opentelemetry.instrumentation.fastapi",
            FastAPIInstrumentor=_FakeFastAPIInstrumentor,
        ),
    )
    monkeypatch.setattr(tracing_module, "find_spec", lambda name: object())
    return _FakeFastAPIInstrumentor


def test_fastapi_app_is_instrumented(fake_fastapi) -> None:
    app = object()

    assert (
        tracing_module.instrument_fastapi_app(
            app, "provider", excluded_urls="/health,/ready"
        )
        is True
    )
    assert _FakeFastAPIInstrumentor.instrumented == [
        {
            "app": app,
            "tracer_provider": "provider",
            "excluded_urls": "/health,/ready",
            "exclude_spans": ["receive", "send"],
        }
    ]


def test_fastapi_not_instrumented_without_provider(fake_fastapi) -> None:
    # Трейсинг выключен — setup_tracing вернул None, серверных спанов нет.
    assert tracing_module.instrument_fastapi_app(object(), None) is False
    assert _FakeFastAPIInstrumentor.instrumented == []


def test_fastapi_empty_excluded_urls_passed_as_none(fake_fastapi) -> None:
    tracing_module.instrument_fastapi_app(object(), "provider", excluded_urls="")
    assert _FakeFastAPIInstrumentor.instrumented[0]["excluded_urls"] is None


def test_fastapi_missing_package_does_not_raise(monkeypatch) -> None:
    monkeypatch.setattr(
        tracing_module, "find_spec", lambda name: None if name.endswith("fastapi") else object()
    )

    assert tracing_module.instrument_fastapi_app(object(), "provider") is False
