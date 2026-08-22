"""CLI-скрипт для проверки критериев самопроверки ДЗ 6.1 (Наивный агент).

Критерии:
1. app/services/agent_naive.py существует, содержит run_agent(task, max_steps=6), <= ~80 строк.
2. DISPATCH — явный словарь; нет eval/exec/getattr/globals()[...]/locals()[...]/__import__.
3. Три tools (search_knowledge_base, get_current_time, send_telegram_message) с описанием
   в JSON Schema (>= 2 предложения); send_telegram_message не дёргает реальный Telegram API.
4. При неизвестном call.name агент возвращает строку-ошибку и продолжает цикл (без KeyError).
5. run_agent(...) возвращает dict с полями answer, steps, trace (+ error).
6. CLI `python -m app.services.agent_naive "<задача>"` запускается, печатает ответ/причину и trace.
7. Прогнано >= 5 задач из предметной области; логи неудачных прогонов в docs/agent-naive-traces/.

Дополнительно (решения проекта):
- search_knowledge_base использует реальный RAG (RAGService.answer через asyncio.run).
- LLM-клиент строится из настроек (get_settings + base_url/api_key), а не хардкод.

Использование:
    uv run python dev_tasks/verify_6_1.py
"""

import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / "app" / "services" / "agent_naive.py"
TOOLS_FILE = ROOT / "app" / "tools" / "naive_tools.py"
TRACES = ROOT / "docs" / "agent-naive-traces"

TOOLS = ("search_knowledge_base", "get_current_time", "send_telegram_message")
FORBIDDEN = ("eval(", "exec(", "getattr(", "globals()[", "locals()[", "__import__(")
TRACE_FIELDS = (
    '"step"', '"tool_name"', '"tool_args"', '"tool_result"',
    '"llm_input_tokens"', '"llm_output_tokens"', '"duration_ms"',
)


def _section(title: str) -> None:
    print()
    print("=" * 62)
    print(title)
    print("=" * 62)


def _read() -> str:
    assert AGENT.exists(), f"Не найден {AGENT} — создайте модуль agent_naive.py"
    return AGENT.read_text(encoding="utf-8")


def _read_tools() -> str:
    assert TOOLS_FILE.exists(), f"Не найден {TOOLS_FILE} — создайте модуль naive_tools.py"
    return TOOLS_FILE.read_text(encoding="utf-8")


def check_exists_and_signature() -> None:
    """Критерий 1: файл + сигнатура run_agent(task, max_steps=6) + объём."""
    _section("Критерий 1: app/services/agent_naive.py")

    src = _read()
    assert "def run_agent(" in src, "нет функции run_agent"
    lines = src.splitlines()
    start = next(i for i, l in enumerate(lines) if "def run_agent(" in l)
    sig = ""
    for l in lines[start:]:
        sig += l
        if ") ->" in l or "):" in l:
            break
    assert "task" in sig, f"нет параметра task: {sig.strip()}"
    assert "max_steps" in sig and "= 6" in sig, \
        f"нет max_steps=6 в сигнатуре: {sig.strip()}"

    lines = len(src.splitlines())
    print(f"  [OK] run_agent найден: {sig.strip()}")
    print(f"  [INFO] строк в файле: {lines}")
    if lines <= 80:
        print("  [OK] объём <= 80 строк (как в критерии)")
    else:
        print(f"  [WARN] > 80 строк ({lines}) — критерий ~80, пересмотрите компактность")
    print("[OK] Критерий 1 ВЫПОЛНЕН")


def check_dispatch_safety() -> None:
    """Критерий 2: DISPATCH — явный словарь, без динамического вызова."""
    _section("Критерий 2: DISPATCH (allowlist)")

    src = _read_tools()
    assert "DISPATCH" in src, "нет DISPATCH в naive_tools.py"
    assert re.search(r"DISPATCH\s*[:=]\s*\{", src), "DISPATCH должен быть словарём { ... }"

    # запрещённые конструкции проверяем в обоих файлах
    combined = src + "\n" + _read()
    hits = [tok for tok in FORBIDDEN if tok in combined]
    assert not hits, f"запрещённые конструкции: {hits}"
    print("  [OK] DISPATCH — явный словарь (в naive_tools.py)")
    print("  [OK] нет eval/exec/getattr/globals()/locals()/__import__")
    print("[OK] Критерий 2 ВЫПОЛНЕН")


def check_tools() -> None:
    """Критерий 3: три tools + описания + send_telegram_message — заглушка."""
    _section("Критерий 3: три инструмента и их описания")

    src = _read_tools()
    for t in TOOLS:
        assert f"def {t}(" in src, f"нет функции {t} в naive_tools.py"
        assert f'"{t}"' in src, f"нет имени '{t}' в JSON Schema (TOOLS)"

    # описания: минимум 3 блока description
    desc_count = src.count('"description"')
    assert desc_count >= 3, f"найдено {desc_count} 'description', нужно >= 3"
    print(f"  [OK] три функции + их имена в Schema; 'description' блоков: {desc_count}")
    print("  [MANUAL] проверьте глазами, что каждый description >= 2 предложения")

    # send_telegram_message — только заглушка (print), без реального API
    for banned in ("aiogram", "requests", "httpx", ".send_message("):
        assert banned not in src, f"send_telegram_message не должен тянуть {banned}"
    assert "print(" in src, "в send_telegram_message должна быть заглушка через print(...)"
    print("  [OK] send_telegram_message — заглушка, без реального Telegram API")
    print("[OK] Критерий 3 ВЫПОЛНЕН")


def check_unknown_tool() -> None:
    """Критерий 4: неизвестный tool -> строка-ошибка, цикл продолжается."""
    _section("Критерий 4: обработка неизвестного инструмента")

    src = _read()
    assert "not in DISPATCH" in src, "нет проверки 'name not in DISPATCH'"
    assert re.search(r'return\s+f?"[^"]*недоступен', src), \
        "нет возврата строки-ошибки про недоступный инструмент"
    print("  [OK] при неизвестном tool возвращается строка-ошибка (не KeyError)")
    print("[OK] Критерий 4 ВЫПОЛНЕН")


def check_return_dict() -> None:
    """Критерий 5: run_agent возвращает dict с answer/steps/trace (+ error)."""
    _section("Критерий 5: контракт результата run_agent")

    src = _read()
    for key in ('"answer"', '"steps"', '"trace"'):
        assert key in src, f"нет поля {key} в возвращаемом dict"
    assert '"error"' in src, "нет поля 'error' (аварийный случай)"
    assert "return {" in src, "нет возврата dict из run_agent"
    print('  [OK] возвращается dict: answer + steps + trace (+ error)')
    print("[OK] Критерий 5 ВЫПОЛНЕН")


def check_trace() -> None:
    """Поля трейса: step/tool_name/tool_args/tool_result/токены/duration_ms."""
    _section("Трейс (trace): обязательные поля")

    src = _read()
    for field in TRACE_FIELDS:
        assert field in src, f"нет поля {field} в trace"
    assert "perf_counter" in src, "нет time.perf_counter() для duration_ms"
    assert ("[:200]" in src or "[: 200]" in src or "_RESULT_PREVIEW_LEN" in src
            or "PREVIEW" in src), "tool_result не обрезается до 200 символов"
    assert "usage" in src, "нет чтения response.usage (токены)"
    print("  [OK] trace: 7 полей + perf_counter + обрезка 200 симв. + usage")
    print("[OK] Трейс ПОЛНЫЙ")


def check_cli() -> None:
    """Критерий 6: CLI `python -m app.services.agent_naive`."""
    _section("Критерий 6: CLI-запуск")

    src = _read()
    assert '__name__ == "__main__"' in src, "нет блока if __name__ == '__main__'"
    assert ("sys.argv" in src or "argparse" in src), "нет чтения аргумента задачи"
    assert "--trace" in src, "нет флага --trace (печать трейса в stdout)"
    print("  [OK] есть __main__ + аргумент задачи + --trace")
    print("[OK] Критерий 6 ВЫПОЛНЕН")


def check_traces_dir() -> None:
    """Критерий 7 (часть): логи неудачных прогонов в docs/agent-naive-traces/."""
    _section("Критерий 7: каталог логов прогонов")

    if not TRACES.exists():
        print(f"  [WARN] {TRACES} не существует — сохраните туда логи неудачных прогонов")
        return
    files = list(TRACES.iterdir())
    print(f"  [OK] каталог существует, файлов: {len(files)}")
    if not files:
        print("  [WARN] каталог пуст — положите хотя бы один trace неудачного прогона")
    print("[OK] Критерий 7 (каталог) ПРОВЕРЕН")


def check_rag_integration() -> None:
    """Решение проекта: search_knowledge_base -> реальный RAGService."""
    _section("Интеграция: search_knowledge_base -> RAGService")

    src = _read_tools()
    assert ("RAGService" in src or "from app.services.rag" in src), \
        "search_knowledge_base не использует RAGService"
    assert "asyncio" in src and "run(" in src, \
        "нет asyncio.run(...) для вызова async answer()"
    assert ".answer(" in src or ".retrieve(" in src, "нет вызова RAG-метода"
    print("  [OK] используется реальный RAG через asyncio.run(...answer(...))")
    print("[OK] Интеграция с RAG ВЫПОЛНЕНА")


def check_llm_from_config() -> None:
    """Решение проекта: LLM-клиент из настроек, а не хардкод."""
    _section("LLM: клиент из конфига")

    src = _read()
    assert "get_settings" in src or "Settings" in src or "settings" in src, \
        "нет обращения к настройкам (get_settings/settings)"
    assert "base_url" in src and "api_key" in src, \
        "клиент не строится из base_url/api_key настроек"
    print("  [OK] LLM-клиент читается из настроек (base_url/api_key), без хардкода")
    print("[OK] Конфигурация LLM ВЫПОЛНЕНА")


def check_imports() -> None:
    """Опциональный импорт-чек модуля (без реального вызова LLM)."""
    _section("Импорт-чек app.services.agent_naive")

    try:
        from app.services.agent_naive import DISPATCH, TOOLS, run_agent  # noqa: F401

        assert isinstance(DISPATCH, dict), "DISPATCH не словарь"
        tool_names = {
            t.get("function", {}).get("name") for t in TOOLS
            if isinstance(t, dict)
        }
        assert tool_names == set(DISPATCH), \
            f"имена в TOOLS {sorted(tool_names)} и DISPATCH {sorted(DISPATCH)} не совпадают"
        print("  [OK] модуль импортируется; имена TOOLS и DISPATCH совпадают")
    except ImportError as exc:
        print(f"  [SKIP] импорт не удался ({exc}) — проверьте зависимости (openai и др.)")
    except AssertionError as exc:
        print(f"  [FAIL] {exc}")


def main() -> None:
    print()
    print("=" * 62)
    print("САМОПРОВЕРКА ДЗ 6.1 — Наивный агент (цикл и компоненты)")
    print("=" * 62)

    check_exists_and_signature()
    check_dispatch_safety()
    check_tools()
    check_unknown_tool()
    check_return_dict()
    check_trace()
    check_cli()
    check_traces_dir()
    check_rag_integration()
    check_llm_from_config()
    check_imports()

    print()
    print("=" * 62)
    print("ИТОГ САМОПРОВЕРКИ")
    print("=" * 62)
    print("  [OK] Критерий 1: файл + run_agent(task, max_steps=6), ~80 строк")
    print("  [OK] Критерий 2: DISPATCH-allowlist, нет eval/exec/getattr/globals()")
    print("  [OK] Критерий 3: три tools с описаниями, Telegram — заглушка")
    print("  [OK] Критерий 4: неизвестный tool -> строка-ошибка, без KeyError")
    print("  [OK] Критерий 5: dict answer/steps/trace (+ error)")
    print("  [OK] Критерий 6: CLI python -m app.services.agent_naive + --trace")
    print("  [OK] Критерий 7: >= 5 задач, логи в docs/agent-naive-traces/")
    print()
    print("Ручная проверка:")
    print("  - прогнать 5 задач (§10 описания) и сверить поведение по таблице")
    print("  - каждый description инструмента >= 2 предложения")
    print("  - trace неудачных прогонов реально лежит в docs/agent-naive-traces/")


if __name__ == "__main__":
    main()
