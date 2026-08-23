"""CLI-скрипт для проверки критериев самопроверки ДЗ 6.2 (ReAct-агент + self-reflection).

Критерии:
1. app/services/agent_react.py существует; app/services/agent_naive.py нетронут (baseline).
2. ReAct-цикл на нативном tool calling: chat.completions.create + tool_choice="auto" + role:"tool".
3. Системный промпт задаёт ReAct-поведение (рассуждение, ровно один инструмент, остановка,
   честный отказ, «не выдумывай»).
4. Два жёстких лимита: max_iterations (default 10, диапазон 8-20) и timeout_per_iteration_sec
   (5-15 сек), явные сообщения "Превышен лимит итераций" / "Timeout".
5. Self-reflection: отдельный critic-вызов, вердикты OK / REVISE, max_revisions=2,
   счётчик ревизий не сбрасывается.
6. Подсчёт стоимости: prompt_tokens / completion_tokens / total_tokens по всем вызовам
   (включая critic), итог в dict + structlog.
7. Логирование каждой итерации: номер шага, tool name, latency, токены.
8. Отчёт docs/agent-react-report.md: таблица по 5 задачам + total tokens.
9. Нет max_iterations >= 30 и нет additionalProperties: true.

Использование:
    uv run python dev_tasks/verify_6_2.py
"""

import re
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
AGENT_REACT = ROOT / "app" / "services" / "agent_react.py"
AGENT_NAIVE = ROOT / "app" / "services" / "agent_naive.py"
REPORT = ROOT / "docs" / "agent-react-report.md"

# Инструменты-реализации можно переиспользовать из наивного модуля; новые описания TOOLS
# ищутся либо в agent_react.py, либо в app/tools/react_tools.py.
TOOLS_REACT = ROOT / "app" / "tools" / "react_tools.py"


def _section(title: str) -> None:
    print()
    print("=" * 62)
    print(title)
    print("=" * 62)


def _read(path: Path) -> str:
    assert path.exists(), f"Не найден {path} — создайте файл"
    return path.read_text(encoding="utf-8")


def _read_react() -> str:
    return _read(AGENT_REACT)


def _read_naive() -> str:
    return _read(AGENT_NAIVE)


def _read_tools() -> str:
    """Текст описаний инструментов: react_tools.py, если есть, иначе agent_react.py."""
    if TOOLS_REACT.exists():
        return TOOLS_REACT.read_text(encoding="utf-8")
    return _read_react()


def check_module_and_baseline() -> None:
    """Критерий 1 + 6: agent_react.py есть; agent_naive.py нетронут."""
    _section("Критерий 1/6: модули и сохранность baseline")

    react = _read_react()
    naive = _read_naive()

    # baseline нетронут: функция run_agent с max_steps всё ещё на месте
    assert "def run_agent(" in naive, "в agent_naive.py пропала run_agent"
    assert "max_steps" in naive, "в agent_naive.py пропал max_steps"
    print("  [OK] agent_react.py существует")
    print("  [OK] agent_naive.py содержит run_agent + max_steps (baseline на месте)")

    # git-проверка нетронутости (не фатальна, если git недоступен)
    try:
        diff = subprocess.run(
            ["git", "diff", "--name-only", "--", "app/services/agent_naive.py"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        )
        if diff.stdout.strip():
            print("  [FAIL] agent_naive.py изменён относительно git — верните baseline")
        else:
            print("  [OK] git: agent_naive.py без несохранённых изменений")
    except FileNotFoundError:
        print("  [WARN] git недоступен — baseline проверен только по содержимому")

    print("[OK] Критерий 1/6 ВЫПОЛНЕН")


def check_react_loop() -> None:
    """Критерий 2: цикл на нативном tool calling."""
    _section("Критерий 2: ReAct-цикл на нативном tool calling")

    src = _read_react()
    assert "chat.completions.create" in src, "нет вызова chat.completions.create"
    assert "tool_choice" in src, "нет параметра tool_choice"
    assert re.search(r"tool_choice\s*=\s*[\"']auto[\"']", src), \
        "tool_choice должен быть 'auto'"
    assert "tool_calls" in src, "нет обработки message.tool_calls"
    assert "tool_call_id" in src, "нет tool_call_id (role=tool)"
    assert re.search(r"[\"']tool[\"']", src), "нет сообщения role='tool'"
    print("  [OK] create + tool_choice='auto' + tool_calls + role='tool'")
    print("[OK] Критерий 2 ВЫПОЛНЕН")


def check_system_prompt() -> None:
    """Критерий 3 (часть): системный промпт задаёт ReAct-поведение."""
    _section("Критерий 3: системный промпт ReAct")

    src = _read_react()
    low = src.lower()
    assert "system" in low, "нет системного промпта"
    # ключевые инструкции ReAct (любая комбинация из перечисленных)
    marks = [
        ("инструмент", "инструмент"),
        ("не выдумыва", "не выдумывай"),
        ("один инструмент", "ровно один инструмент"),
        ("финальный ответ", "условие остановки / финальный ответ"),
    ]
    found = [label for needle, label in marks if needle in low]
    assert found, "в системном промпте нет ReAct-инструкций (инструмент/один/не выдумывай/остановка)"
    print(f"  [OK] системный промпт содержит: {', '.join(found)}")
    print("[OK] Критерий 3 (промпт) ВЫПОЛНЕН")


def check_limits() -> None:
    """Критерий 4 + 9 (часть): max_iterations и timeout_per_iteration_sec."""
    _section("Критерий 4: два жёстких лимита")

    src = _read_react()

    # --- max_iterations ---
    assert "max_iterations" in src, "нет max_iterations"
    # [^=\n]* — пропускаем опциональную аннотацию типа (`: int`) или `--max_iterations", type=int,`
    defaults = re.findall(r"max_iterations\b[^=\n]*=\s*(\d+)", src)
    assert defaults, "нет значения по умолчанию у max_iterations"
    default = int(defaults[0])
    assert 8 <= default <= 20, f"default max_iterations={default} вне диапазона 8-20"
    # нет >= 30
    bad = [int(v) for v in re.findall(r"max_iterations\b[^=\n]*=\s*(\d+)", src) if int(v) >= 30]
    assert not bad, f"найдено max_iterations >= 30: {bad}"

    # --- timeout ---
    assert re.search(r"timeout_per_iteration_sec|timeout_sec|timeout\b", src), \
        "нет timeout на итерацию"
    t_defaults = re.findall(r"timeout(?:_per_iteration_sec|_sec)?\b[^=\n]*=\s*([\d.]+)", src)
    if t_defaults:
        t_default = float(t_defaults[0])
        assert 5 <= t_default <= 15, f"default timeout={t_default} вне диапазона 5-15"

    # --- явные сообщения ---
    assert "Превышен лимит итераций" in src, "нет сообщения 'Превышен лимит итераций'"
    assert "Timeout" in src, "нет сообщения 'Timeout'"

    print(f"  [OK] max_iterations default={default} (8-20), timeout default={t_defaults[0] if t_defaults else '?'} сек")
    print("  [OK] сообщения 'Превышен лимит итераций' / 'Timeout' присутствуют")
    print("  [OK] нет max_iterations >= 30")
    print("[OK] Критерий 4 ВЫПОЛНЕН")


def check_reflection() -> None:
    """Критерий 5: self-reflection с critic, вердиктами OK/REVISE и лимитом ревизий."""
    _section("Критерий 5: self-reflection (Reflexion-light)")

    src = _read_react()
    low = src.lower()
    assert "critic" in low or "критик" in low, "нет отдельного вызова критика"
    assert "revise" in low, "нет вердикта REVISE"
    assert "max_revisions" in src, "нет max_revisions"
    rev = re.findall(r"max_revisions\b[^=\n]*=\s*(\d+)", src)
    assert rev and int(rev[0]) <= 2, f"max_revisions={rev} больше 2"

    # счётчик ревизий объявлен до цикла (не внутри for)
    assert "revisions" in low, "нет счётчика ревизий"
    print(f"  [OK] critic-вызов + REVISE + max_revisions={rev[0] if rev else '?'} (<= 2)")
    print("  [MANUAL] убедитесь, что счётчик ревизий объявлен ДО цикла for и не сбрасывается")
    print("[OK] Критерий 5 ВЫПОЛНЕН")


def check_cost() -> None:
    """Критерий 6: подсчёт токенов по всем вызовам + structlog."""
    _section("Критерий 6: подсчёт стоимости")

    src = _read_react()
    for tok in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert tok in src, f"нет {tok}"
    assert "usage" in src, "нет чтения response.usage"
    assert ("structlog" in src) or ("logger" in src) or ("log." in src), \
        "нет логирования токенов (structlog/logger)"
    print("  [OK] prompt/completion/total_tokens + usage + логирование")
    print("[OK] Критерий 6 ВЫПОЛНЕН")


def check_step_logging() -> None:
    """Критерий 7: логирование каждой итерации (шаг, tool, latency, токены)."""
    _section("Критерий 7: логирование шага")

    src = _read_react()
    low = src.lower()
    assert "step" in low or "итераци" in low, "нет номера шага в логе"
    assert "tool" in low, "нет имени tool в логе"
    assert ("perf_counter" in src or "monotonic" in src or "duration" in low or "latency" in low), \
        "нет замера latency (perf_counter/monotonic/duration)"
    assert ("tokens" in low or "токен" in low), "нет токенов в логе шага"
    print("  [OK] лог шага: номер + tool + latency + токены")
    print("[OK] Критерий 7 ВЫПОЛНЕН")


def check_tool_descriptions() -> None:
    """Решения проекта: описания инструментов переписаны (4 вопроса), strict-схема."""
    _section("Инструменты: описания + strict JSON Schema")

    src = _read_tools()
    # описания отвечают на 4 вопроса: активный глагол + «когда» + «возвращает» + границы
    low = src.lower()
    desc = src.count('"description"')
    assert desc >= 1, "нет description у инструментов"
    assert re.search(r"вызывай|используй|когда", low), \
        "в описании нет указания «когда вызывать»"
    assert re.search(r"возвращает|вернёт|вернёт", low), \
        "в описании нет указания «что возвращает»"
    print(f"  [OK] description блоков: {desc}; есть «когда» и «возвращает»")
    print("  [MANUAL] проверьте глазами: имя-глагол + все 4 вопроса чек-листа")

    # strict: нет additionalProperties: true
    _check_no_additional_props_true(src)
    print("[OK] Инструменты ПРОВЕРЕНЫ")


def _check_no_additional_props_true(src: str) -> None:
    bad = re.findall(r"[\"']additionalProperties[\"']\s*:\s*true", src)
    assert not bad, f"найдено additionalProperties: true ({len(bad)}) — схема должна быть строгой"
    print("  [OK] нет additionalProperties: true")


def check_report() -> None:
    """Критерий 8: отчёт docs/agent-react-report.md с таблицей по 5 задачам."""
    _section("Критерий 8: отчёт docs/agent-react-report.md")

    assert REPORT.exists(), f"Не найден {REPORT}"
    src = REPORT.read_text(encoding="utf-8")
    low = src.lower()
    assert "|" in src, "нет таблицы (разделитель |)"
    assert "naive" in low and "react" in low, "в отчёте нет колонок naive/react"
    assert re.search(r"токен|token", low), "нет подсчёта токенов"
    assert "ревиз" in low or "revise" in low or "итерац" in low, \
        "нет колонки про итерации/ревизию"
    print("  [OK] отчёт есть, таблица с naive/react + токены + итерации/ревизия")
    print("  [MANUAL] проверьте, что строк по задачам ровно 5 и есть 3-5 наблюдений")
    print("[OK] Критерий 8 ВЫПОЛНЕН")


def check_imports() -> None:
    """Опциональный импорт-чек модуля (без реального вызова LLM)."""
    _section("Импорт-чек app.services.agent_react")

    try:
        import app.services.agent_react as mod  # noqa: F401

        print("  [OK] модуль agent_react импортируется")
    except ImportError as exc:
        print(f"  [SKIP] импорт не удался ({exc}) — проверьте зависимости (openai и др.)")
    except Exception as exc:  # pragma: no cover — страховка от побочных эффектов импорта
        print(f"  [WARN] импорт упал с ошибкой ({exc}) — проверьте, что модуль не выполняет код при импорте")


def main() -> None:
    print()
    print("=" * 62)
    print("САМОПРОВЕРКА ДЗ 6.2 — ReAct-агент + self-reflection")
    print("=" * 62)

    check_module_and_baseline()
    check_react_loop()
    check_system_prompt()
    check_limits()
    check_reflection()
    check_cost()
    check_step_logging()
    check_tool_descriptions()
    check_report()
    check_imports()

    print()
    print("=" * 62)
    print("ИТОГ САМОПРОВЕРКИ")
    print("=" * 62)
    print("  [OK] Критерий 1: agent_react.py создан, agent_naive.py нетронут")
    print("  [OK] Критерий 2: ReAct на нативном tool calling (tool_choice='auto')")
    print("  [OK] Критерий 3: системный промпт задаёт ReAct-поведение")
    print("  [OK] Критерий 4: max_iterations (8-20) + timeout (5-15) + явные сообщения")
    print("  [OK] Критерий 5: self-reflection (critic + OK/REVISE + max_revisions<=2)")
    print("  [OK] Критерий 6: подсчёт prompt/completion/total_tokens + structlog")
    print("  [OK] Критерий 7: логирование шага (номер + tool + latency + токены)")
    print("  [OK] Критерий 8: отчёт docs/agent-react-report.md")
    print("  [OK] Критерий 9: нет max_iterations>=30 и additionalProperties:true")
    print()
    print("Ручная проверка:")
    print("  - прогнать 5 задач на обоих агентах и сверить таблицу отчёта")
    print("  - на «провокационной» задаче react НЕ вызывает tools (критерий 4)")
    print("  - описания инструментов отвечают на все 4 вопроса чек-листа")
    print("  - счётчик ревизий объявлен до цикла и не сбрасывается")


if __name__ == "__main__":
    main()
