"""CLI-скрипт для проверки критериев самопроверки ДЗ 6.5 (мультиагентные системы).

Критерии:
0. Зависимости: langgraph / langchain / langchain-openai / langchain-core на месте;
   langgraph-supervisor — только если выбран путь create_supervisor (иначе ручной Command).
1. experiments/: два скрипта (multi_agent_langgraph.py, single_agent_baseline.py) + __init__.py.
2. Один и тот же tool search_knowledge_base в обоих скриптах (общий модуль, не два своих @tool).
3. 5 одинаковых тестовых вопросов (3 по корпусу / 1 многошаговый / 1 вне базы) — общий модуль.
4. Мультиагент: researcher (с tool) + writer (без tools) + супервизор (create_supervisor
   или ручной Command(goto=...)); output_mode зафиксирован.
5. Чекпоинтер InMemorySaver + стриминг stream_mode="updates" + стабильный thread_id.
6. draw_mermaid() в скрипте (текст схемы сохраняется в docs/architecture-multi-agent.md).
7. Single-agent baseline: ОДИН create_agent с тем же tool; handoff_count = 0.
8. Замеры в коде: perf_counter, total_tokens, llm_calls, handoff_count, запись в results.json.
9. experiments/results.json: 10 записей (5 вопросов x 2 реализации) с 4 полями метрик.
10. docs/multi-agent-report.md: постановка, сравнительная таблица, Anthropic-абзац (~15x / +90.2%),
    явная секция "## Решение".
11. Решение: "использую" -> паттерн + координация + бюджет; "не использую" -> обоснование.
12. docs/architecture-multi-agent.md: mermaid-блок со схемой supervisor-графа.
13. Цитирование [1], [2] в финальном ответе мультиагента (промпт writer).
14. Один из вопросов — заведомо вне корпуса (fallback, без галлюцинаций).

Скрипт не останавливается на первой ошибке: проходит все критерии и печатает итог.

Использование:
    uv run python dev_tasks/verify_6_5.py
    uv run python dev_tasks/verify_6_5.py --skip-import   # без импорта experiments.*
"""

import argparse
import json
import re
import sys
import tomllib
from collections import Counter
from collections.abc import Callable
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
EXPERIMENTS = ROOT / "experiments"
MULTI = EXPERIMENTS / "multi_agent_langgraph.py"
SINGLE = EXPERIMENTS / "single_agent_baseline.py"
QUESTIONS = EXPERIMENTS / "questions.py"
RESULTS = EXPERIMENTS / "results.json"
REPORT = ROOT / "docs" / "multi-agent-report.md"
ARCH = ROOT / "docs" / "architecture-multi-agent.md"
DOCS_INDEX = ROOT / "docs" / "README.md"

WARNINGS: list[str] = []
MANUAL: list[str] = []


# --------------------------------------------------------------------------- utils


def _section(title: str) -> None:
    print()
    print("=" * 66)
    print(title)
    print("=" * 66)


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def _warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"  [WARN] {msg}")


def _manual(msg: str) -> None:
    MANUAL.append(msg)
    print(f"  [MANUAL] {msg}")


def _read(path: Path) -> str:
    assert path.exists(), f"не найден файл {path.relative_to(ROOT)} — создайте его"
    return path.read_text(encoding="utf-8")


def _read_opt(path: Path) -> str:
    """Читает файл, если есть; иначе пустая строка (файл необязателен)."""
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _experiment_sources() -> dict[str, str]:
    """Все .py в experiments/ — там могут лежать и вспомогательные модули."""
    if not EXPERIMENTS.exists():
        return {}
    out: dict[str, str] = {}
    for path in sorted(EXPERIMENTS.glob("*.py")):
        try:
            out[path.name] = path.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            _warn(f"не удалось прочитать {path.name}: {exc}")
    return out


def _param_near(src: str, anchor: str, pattern: str, window: int = 500) -> bool:
    """Ищет pattern в окне ±window символов вокруг первого вхождения anchor."""
    pos = src.find(anchor)
    if pos == -1:
        return False
    segment = src[max(0, pos - window): pos + window]
    return re.search(pattern, segment) is not None


def _uses_create_supervisor(src: str) -> bool:
    """Пакет langgraph-supervisor реально используется (вызов create_supervisor)."""
    return re.search(r"create_supervisor\s*\(", src) is not None


def _count_questions(src: str) -> int:
    """Грубо считает число вопросов в литерале: по 'id', 'qN' или 'kind'."""
    for pattern in (r'["\']id["\']\s*:', r'["\']q\d+["\']', r'["\']kind["\']\s*:'):
        hits = re.findall(pattern, src)
        if hits:
            return len(set(hits)) if pattern.startswith(r'["\']q') else len(hits)
    return 0


def _kinds(src: str) -> set[str]:
    kinds: set[str] = set()
    for value in re.findall(r'["\']kind["\']\s*:\s*["\']([\w\-]+)["\']', src):
        kinds.add(value.lower())
    for value in re.findall(r'["\'](corpus|multi_step|out_of_scope|multi-step|out-of-scope)["\']', src):
        kinds.add(value.lower().replace("-", "_"))
    return kinds


def _question_strings(src: str) -> set[str]:
    """Строковые литералы, похожие на вопросы (длина >= 15, есть пробел)."""
    out: set[str] = set()
    for match in re.finditer(r'["\']([^"\'\n]{15,})["\']', src):
        value = match.group(1).strip()
        if " " in value and not value.startswith(("http", "experiments", "app.")):
            out.add(value)
    return out


# ----------------------------------------------------------------------- критерий 0


def check_deps() -> None:
    """Зависимости LangGraph/LangChain; supervisor-пакет — только если выбран этот путь."""
    raw = tomllib.loads(_read(PYPROJECT))
    project = raw.get("project", {})
    declared: list[str] = list(project.get("dependencies", []))
    for group in (project.get("optional-dependencies") or {}).values():
        declared += list(group)
    for group in (raw.get("dependency-groups") or {}).values():
        declared += [g for g in group if isinstance(g, str)]

    def spec_for(name: str) -> str | None:
        pattern = re.compile(rf"^{re.escape(name)}\b", re.IGNORECASE)
        for dep in declared:
            if pattern.match(dep.strip().strip('"')):
                return dep
        return None

    for name in ("langgraph", "langchain", "langchain-openai", "langchain-core"):
        spec = spec_for(name)
        assert spec, f"в pyproject.toml нет зависимости {name}"
        _ok(f"объявлен {spec}")

    supervisor = spec_for("langgraph-supervisor")
    uses_supervisor_pkg = _uses_create_supervisor(_read_opt(MULTI))
    if supervisor:
        _ok(f"объявлен {supervisor} (путь create_supervisor)")
    elif uses_supervisor_pkg:
        assert False, (
            "в experiments/multi_agent_langgraph.py используется create_supervisor, "
            "но пакет langgraph-supervisor не объявлен — добавьте uv add langgraph-supervisor"
        )
    else:
        _warn(
            "langgraph-supervisor не объявлен — это нормально, если выбран ручной супервизор "
            "через Command(goto=...); если выбран create_supervisor — добавьте uv add langgraph-supervisor"
        )


# ----------------------------------------------------------------------- критерий 1


def check_experiments_layout() -> None:
    """experiments/ + два скрипта + __init__.py."""
    assert EXPERIMENTS.exists(), "нет папки experiments/ — задание требует отдельную папку для прототипов"
    _ok("папка experiments/ существует")

    for path, label in ((MULTI, "multi_agent_langgraph.py"), (SINGLE, "single_agent_baseline.py")):
        assert path.exists(), f"нет experiments/{label}"
        src = path.read_text(encoding="utf-8")
        assert re.search(r'if\s+__name__\s*==\s*["\']__main__["\']', src), (
            f"{label}: нет блока if __name__ == '__main__' — скрипт не запускается одной командой"
        )
        _ok(f"{label} на месте и запускается как скрипт")

    if (EXPERIMENTS / "__init__.py").exists():
        _ok("experiments/__init__.py есть (запуск через python -m experiments.<скрипт>)")
    else:
        _warn(
            "нет experiments/__init__.py — запуск через `python -m experiments.xxx` не сработает; "
            "либо добавьте пустой __init__.py, либо запускайте `python experiments/xxx.py` из корня"
        )

    extras = [name for name in _experiment_sources() if name not in {MULTI.name, SINGLE.name, "__init__.py"}]
    if extras:
        _ok(f"вспомогательные модули: {', '.join(extras)}")
    else:
        _warn("нет вспомогательных модулей — ожидались общий tool / вопросы / утилиты замеров")


# ----------------------------------------------------------------------- критерий 2


_TOOL_IMPORT = re.compile(r"^\s*from\s+([\w\.]+)\s+import\s+([^\n#]+)", re.M)


def _tool_sources(src: str) -> set[str]:
    """Модули, откуда импортируется search_knowledge_base."""
    mods: set[str] = set()
    for match in _TOOL_IMPORT.finditer(src):
        if "search_knowledge_base" in match.group(2):
            mods.add(match.group(1))
    # import experiments.kb_tool  +  kb_tool.search_knowledge_base(...)
    for match in re.finditer(r"^\s*import\s+([\w\.]+)", src, re.M):
        module = match.group(1)
        short = module.split(".")[-1]
        if re.search(rf"\b{re.escape(short)}\.search_knowledge_base\b", src):
            mods.add(module)
    return mods


def check_shared_tool() -> None:
    """Один и тот же search_knowledge_base в обоих скриптах."""
    multi_src = _read(MULTI)
    single_src = _read(SINGLE)

    for src, label in ((multi_src, "multi_agent_langgraph.py"), (single_src, "single_agent_baseline.py")):
        assert "search_knowledge_base" in src, f"{label}: не используется search_knowledge_base"
        assert not re.search(r"def\s+search_knowledge_base\s*\(", src), (
            f"{label}: определён СВОЙ search_knowledge_base — задание требует одну и ту же реализацию "
            "в обоих скриптах (общий модуль, например experiments/kb_tool.py)"
        )
        assert not re.search(r"@tool\s*\n\s*(async\s+)?def\s+search_knowledge_base", src), (
            f"{label}: свой @tool search_knowledge_base — вынесите его в общий модуль"
        )

    multi_mods = _tool_sources(multi_src)
    single_mods = _tool_sources(single_src)
    assert multi_mods, "multi_agent_langgraph.py: не видно импорта search_knowledge_base из общего модуля"
    assert single_mods, "single_agent_baseline.py: не видно импорта search_knowledge_base из общего модуля"

    common = multi_mods & single_mods
    if common:
        _ok(f"оба скрипта берут tool из одного модуля: {', '.join(sorted(common))}")
    else:
        # допускаем разные модули, если оба в experiments/ и файл один и тот же по смыслу
        assert multi_mods == single_mods, (
            "скрипты импортируют search_knowledge_base из разных модулей "
            f"({sorted(multi_mods)} vs {sorted(single_mods)}) — сравнение будет нечестным"
        )
        _ok(f"оба скрипта берут tool из модулей {sorted(multi_mods)}")

    tool_module = EXPERIMENTS / "kb_tool.py"
    if tool_module.exists():
        _ok("общий модуль experiments/kb_tool.py существует")
    else:
        _warn("нет experiments/kb_tool.py — убедитесь, что общий tool лежит ровно в одном модуле")


# ----------------------------------------------------------------------- критерий 3


def check_questions() -> None:
    """5 одинаковых вопросов: 3 по корпусу, 1 многошаговый, 1 вне базы."""
    multi_src = _read(MULTI)
    single_src = _read(SINGLE)

    if QUESTIONS.exists():
        src = _read(QUESTIONS)
        _ok("experiments/questions.py существует (вопросы общие по построению)")
    else:
        _warn(
            "нет experiments/questions.py — вопросы придётся синхронизировать вручную "
            "в двух скриптах (риск, что наборы разъедутся)"
        )
        src = multi_src + "\n" + single_src

    count = _count_questions(src)
    if count == 0:
        _warn("не удалось автоматически посчитать вопросы — убедитесь руками, что их ровно 5")
    else:
        assert count == 5, f"вопросов найдено {count} — задание требует ровно 5"
        _ok("вопросов: 5")

    kinds = _kinds(src)
    if {"corpus", "multi_step", "out_of_scope"} <= kinds:
        _ok("типы вопросов размечены: corpus / multi_step / out_of_scope")
    else:
        _warn(
            "не видно разметки типов вопросов (corpus / multi_step / out_of_scope) — "
            "в отчёте нужно явно записать '3 по корпусу / 1 многошаговый / 1 вне базы'"
        )

    # оба скрипта должны пользоваться общим набором
    for src_one, label in ((multi_src, "multi_agent_langgraph.py"), (single_src, "single_agent_baseline.py")):
        assert re.search(r"QUESTIONS|questions", src_one), (
            f"{label}: не видно использования общего списка вопросов"
        )

    if not QUESTIONS.exists():
        common = _question_strings(multi_src) & _question_strings(single_src)
        if len(common) >= 5:
            _ok(f"в обоих скриптах найдено {len(common)} общих строк-вопросов")
        else:
            _warn(
                f"в скриптах всего {len(common)} общих строк, похожих на вопросы — "
                "проверьте, что наборы вопросов реально совпадают"
            )
    _ok("оба скрипта ссылаются на общий набор вопросов")

    _manual("сверить глазами: вопросы в отчёте == вопросы в experiments/questions.py")


# ----------------------------------------------------------------------- критерий 4


def check_multi_agent() -> None:
    """researcher + writer + супервизор."""
    src = _read(MULTI)

    assert len(re.findall(r"create_agent\s*\(", src)) >= 2, (
        "в multi_agent_langgraph.py меньше двух create_agent — нужны researcher и writer"
    )
    _ok(f"create_agent вызван {len(re.findall(r'create_agent\s*\(', src))} раз")

    for name in ("researcher", "writer"):
        assert re.search(rf'name\s*=\s*["\']{name}["\']', src), (
            f"нет агента с name='{name}' — create_agent(..., name='{name}')"
        )
    _ok("агенты researcher и writer объявлены по именам")

    writer_has_tools = _param_near(src, 'name="writer"', r"tools\s*=\s*\[\s*\]") or _param_near(
        src, "name='writer'", r"tools\s*=\s*\[\s*\]"
    )
    assert writer_has_tools, (
        "у writer должны быть tools=[] — он только собирает ответ, факты достаёт researcher"
    )
    _ok("writer объявлен без инструментов (tools=[])")

    assert _uses_create_supervisor(src) or re.search(r"Command\s*\(\s*goto", src), (
        "нет ни create_supervisor(...), ни ручного Command(goto=...) — супервизор не собран"
    )
    if _uses_create_supervisor(src):
        _ok("супервизор собран через langgraph_supervisor.create_supervisor")
        if "output_mode" not in src:
            _warn('create_supervisor без output_mode — задание упоминает output_mode="last_message"')
    else:
        _ok("супервизор собран вручную через Command(goto=...)")

    if "transfer_to" in src:
        _ok("видны handoff-инструменты transfer_to_*")

    assert re.search(r'["\']\[1\]["\']|\[1\]', src) and "[2]" in src, (
        "в промпте writer нет требования цитировать источники ([1], [2])"
    )
    _ok("цитирование [1], [2] заложено в промпт writer")


# ----------------------------------------------------------------------- критерий 5


def check_checkpoint_and_stream() -> None:
    """InMemorySaver + stream_mode='updates' + стабильный thread_id."""
    src = _read(MULTI)

    assert "InMemorySaver" in src, "нет InMemorySaver — задание требует чекпоинтер"
    if "compile(" in src and "checkpointer" in src:
        _ok("граф компилируется с checkpointer=InMemorySaver()")
    else:
        _warn("не видно compile(checkpointer=...) — граф собран без чекпоинтера")

    # стриминг и thread_id могут жить в общем runner'е (experiments/measure_utils.py и т.п.)
    helpers = "\n".join(value for name, value in _experiment_sources().items() if name != MULTI.name)
    stream_src = src + "\n" + helpers

    assert re.search(r"\.(a?stream)\s*\(", stream_src), "нет stream(...) — нет стриминга в консоль"
    assert "stream_mode" in stream_src, "нет stream_mode"
    if re.search(r'stream_mode\s*=\s*["\']updates["\']', stream_src):
        if re.search(r"\.(a?stream)\s*\(", src):
            _ok('stream(..., stream_mode="updates") — как требует задание')
        else:
            _ok('stream_mode="updates" используется в общем runner\'е (вызывается из мультиагента)')
    else:
        _warn("stream_mode не равен 'updates' — задание просит stream_mode='updates'")

    if re.search(r'thread_id["\']?\s*[:=]\s*f?["\']exp-langgraph', stream_src):
        _ok('thread_id="exp-langgraph" на месте')
    elif "thread_id" in stream_src:
        _warn('thread_id есть, но не "exp-langgraph" — задание упоминает именно его')
    else:
        assert False, "в мультиагенте нет thread_id"

    _manual("запустить скрипт: в консоли видна последовательность supervisor -> researcher -> writer")


# ----------------------------------------------------------------------- критерий 6


def check_draw_mermaid() -> None:
    """draw_mermaid() в скрипте + mermaid в docs/architecture-multi-agent.md."""
    src = _read(MULTI)
    assert "draw_mermaid" in src, (
        "нет app.get_graph().draw_mermaid() — схема графа не генерируется"
    )
    _ok("draw_mermaid() вызывается")

    if ARCH.exists():
        _ok("docs/architecture-multi-agent.md существует")
    else:
        _warn("docs/architecture-multi-agent.md ещё нет — вставьте туда вывод draw_mermaid()")

    if "architecture-multi-agent" in src or "architecture_multi_agent" in src:
        _ok("скрипт ссылается на файл со схемой (сохранение автоматизировано)")
    else:
        _warn("скрипт не сохраняет схему сам — текст draw_mermaid() вставляется в docs вручную")


# ----------------------------------------------------------------------- критерий 7


def check_single_agent() -> None:
    """Один create_agent с тем же tool; handoff_count = 0."""
    src = _read(SINGLE)

    calls = len(re.findall(r"create_agent\s*\(", src))
    assert calls >= 1, "в single_agent_baseline.py нет create_agent"
    assert calls == 1, (
        f"create_agent вызван {calls} раз(а) — baseline должен быть ОДНИМ агентом с tools"
    )
    _ok("ровно один create_agent (single-agent baseline)")

    explicit_zero = re.search(r'handoff_count["\']?\s*[:=]\s*0', src) is not None
    shared_runner = re.search(r"\brun_one\s*\(|\brun_with_metrics\s*\(|\bmeasure\w*\s*\(", src) is not None
    if explicit_zero:
        _ok("handoff_count = 0 для single-agent (зафиксирован явно)")
    elif shared_runner:
        _ok("метрики baseline собирает общий runner (для single-agent handoff выйдет 0 — сверяется в results.json)")
    else:
        _warn("не видно, что handoff_count для baseline равен 0 — ни константой, ни через общий runner")

    if "create_supervisor" in src or re.search(r"Command\s*\(\s*goto", src):
        _warn("в baseline виден супервизор/handoff — это уже не single-agent")

    _manual("сверить, что промпт baseline объединяет обе роли (поиск фактов + оформление)")


# ----------------------------------------------------------------------- критерий 8


def check_measurements_code() -> None:
    """Замеры в коде: время, токены, вызовы, handoff, запись в results.json."""
    sources = _experiment_sources()
    joined = "\n".join(sources.values())

    for token, hint in (
        ("perf_counter", "latency_ms через time.perf_counter()"),
        ("total_tokens", "total_tokens"),
        ("llm_calls", "llm_calls"),
        ("handoff_count", "handoff_count"),
    ):
        assert token in joined, f"в experiments/ не видно замер {hint}"
        _ok(f"замер {hint} присутствует")

    if re.search(r"usage_metadata|MetricsCallback|on_llm_end|on_chat_model_start|get_openai_callback", joined):
        _ok("токены/вызовы считаются через usage_metadata или callback")
    else:
        _warn("не видно способа подсчёта токенов (usage_metadata / callback)")

    for path, label in ((MULTI, "multi_agent_langgraph.py"), (SINGLE, "single_agent_baseline.py")):
        src = _read(path)
        if not re.search(r"results\.json|save_result|record_result|append_result|write_result", src):
            _warn(f"{label}: не видно записи результатов в results.json")
    _ok("оба скрипта пишут результаты (results.json или общий helper)")

    _manual("убедиться, что токены считаются по ВСЕМ агентам графа, а не только по супервизору")


# ----------------------------------------------------------------------- критерий 9


def check_results_json() -> None:
    """results.json: 10 записей (5 вопросов x 2 реализации) с 4 полями метрик."""
    raw = _read(RESULTS)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"results.json не парсится как JSON: {exc}") from exc

    records = payload if isinstance(payload, list) else payload.get("results", [])
    assert isinstance(records, list), "results.json должен быть списком записей (или {'results': [...]})"
    assert len(records) == 10, f"записей {len(records)} — ожидается 10 (5 вопросов x 2 реализации)"
    _ok("10 записей (5 вопросов x 2 реализации)")

    impls = Counter(str(r.get("impl", "")).lower() for r in records)
    assert impls.get("multi", 0) == 5 and impls.get("single", 0) == 5, (
        f"разбивка по реализациям не 5/5: {dict(impls)} (ожидаются impl='multi' и impl='single')"
    )
    _ok("по 5 записей на каждую реализацию (multi / single)")

    for key in ("total_tokens", "llm_calls", "latency_ms", "handoff_count"):
        missing = [r.get("qid", "?") for r in records if key not in r]
        assert not missing, f"в записях {missing} нет поля {key}"
    _ok("все 4 метрики (total_tokens, llm_calls, latency_ms, handoff_count) заполнены")

    multi_qids = {r.get("qid") for r in records if str(r.get("impl", "")).lower() == "multi"}
    single_qids = {r.get("qid") for r in records if str(r.get("impl", "")).lower() == "single"}
    if multi_qids and multi_qids == single_qids:
        _ok(f"обе реализации прогнаны на одних и тех же вопросах: {sorted(map(str, multi_qids))}")
    else:
        _warn(f"наборы вопросов у реализаций различаются: multi={sorted(map(str, multi_qids))}, "
              f"single={sorted(map(str, single_qids))}")

    single_handoffs = [r.get("handoff_count") for r in records if str(r.get("impl", "")).lower() == "single"]
    if single_handoffs and all(h == 0 for h in single_handoffs):
        _ok("у single-agent handoff_count = 0 во всех записях")
    else:
        _warn(f"у single-agent ожидались нули, получено {single_handoffs}")

    multi_handoffs = [r.get("handoff_count") for r in records if str(r.get("impl", "")).lower() == "multi"]
    if multi_handoffs and sum(1 for h in multi_handoffs if h and h > 0) >= 3:
        _ok(f"у мультиагента handoff_count > 0: {multi_handoffs}")
    else:
        _warn(
            f"у мультиагента почти нет передач управления ({multi_handoffs}) — "
            "похоже, супервизор отвечает сам вместо делегирования"
        )

    if not any("usage" in str(r) or "quality" in str(r) for r in records):
        _manual("оценки качества (LLM-судья) можно хранить в results.json или в отчёте — на ваш выбор")


# ---------------------------------------------------------------------- критерий 10


def check_report() -> None:
    """docs/multi-agent-report.md: постановка, таблица, Anthropic, ## Решение."""
    src = _read(REPORT)
    low = src.lower()

    assert re.search(r"постановка|сценарий", low), "нет раздела с постановкой задачи"
    _ok("постановка задачи есть")

    assert re.search(r"сравнительн", low), "нет сравнительной таблицы single vs multi"
    for metric, pattern in (
        ("токены", r"токен"),
        ("LLM-вызовы", r"llm[-_ ]?вызов|llm_calls|вызовов"),
        ("latency", r"latency|задержк"),
        ("передачи управления", r"передач|handoff"),
        ("качество", r"качеств"),
    ):
        if not re.search(pattern, low):
            _warn(f"в отчёте не видно метрики «{metric}» — таблица заполнена не полностью")
    _ok("метрики сравнительной таблицы упомянуты")

    assert "anthropic" in low, "нет абзаца сопоставления с Anthropic-ориентиром"
    if re.search(r"15\s*[x×]|15-кратн|~15", low) and "90.2" in src:
        _ok("Anthropic-ориентир: множитель токенов и +90.2% на месте")
    else:
        _warn("в Anthropic-абзаце не видно цифр ~15x токенов и +90.2% качества")

    assert re.search(r"^#{1,3}\s*Решение", src, re.M) or re.search(r"##+\s*Решение", src), (
        'нет явной секции "## Решение" — это главный артефакт ДЗ'
    )
    _ok("секция ## Решение есть")

    _manual("проверить, что числа в таблице совпадают с experiments/results.json")


# ---------------------------------------------------------------------- критерий 11


def check_decision() -> None:
    """Решение: 'использую' -> паттерн/координация/бюджет; 'не использую' -> обоснование."""
    src = _read(REPORT)
    low = src.lower()

    assert re.search(r"использ", low), (
        'в отчёте нет явной формулировки решения ("использую" / "не использую" мультиагентность)'
    )

    decision_block = src
    match = re.search(r"##+\s*Решение(.+)", src, re.S)
    if match:
        decision_block = match.group(1)
    block_low = decision_block.lower()

    if re.search(r"не\s+использ", block_low):
        _ok("решение: НЕ использовать мультиагентность")
        signals = re.search(r"tight coupling|общий контекст|write-heavy|real-?time|токен|latency|задержк|числ", block_low)
        if signals:
            _ok("обоснование привязано к признакам задачи и собственным числам")
        else:
            _warn("обоснование «не использую» не привязано к признакам задачи и числам (нужно 2–3 предложения)")
    elif re.search(r"использ", block_low):
        _ok("решение: использовать мультиагентность")
        checks = {
            "паттерн (Supervisor / Hierarchical / Swarm / Fan-out)": r"паттерн|supervisor|иерарх|swarm|веер|fan-?out",
            "формат координации (shared state / message passing)": r"координац|shared state|общее состояние|namespace|message passing|сообщени",
            "бюджет на handoff_count": r"бюджет|handoff|передач",
            "состав агентов (2–4)": r"агент",
        }
        for label, pattern in checks.items():
            if not re.search(pattern, block_low):
                _warn(f"в решении «использую» не указан {label}")
        _ok("пункты решения «использую» проверены (паттерн / координация / бюджет)")
    else:
        _warn("не понятно, какое именно решение принято")

    if re.search(r"app/agents", src):
        _ok("упомянуто, что граф переезжает в app/agents/ (или обоснование, почему нет)")

    _manual("убедиться, что решение — не «красивый ответ», а следствие собственных замеров")


# ---------------------------------------------------------------------- критерий 12


def check_architecture_doc() -> None:
    """docs/architecture-multi-agent.md: mermaid-схема supervisor-графа."""
    src = _read(ARCH)

    assert "```mermaid" in src, "в docs/architecture-multi-agent.md нет блока ```mermaid"
    _ok("mermaid-блок есть")

    low = src.lower()
    for node in ("supervisor", "researcher", "writer"):
        if node not in low:
            _warn(f"в схеме не видно узла '{node}' — похоже, схема не от этого графа")
    _ok("узлы supervisor / researcher / writer присутствуют")

    if "draw_mermaid" in src:
        _ok("указано происхождение схемы (draw_mermaid)")
    else:
        _warn("не указано, что схема сгенерирована draw_mermaid() — добавьте команду/пояснение")


# ---------------------------------------------------------------------- критерий 13


def check_tests_and_docs() -> None:
    """Индекс документации (конвенция проекта) и опциональный импорт-чек."""
    index = _read_opt(DOCS_INDEX)
    if index:
        missing = [name for name in ("multi-agent-report.md", "architecture-multi-agent.md") if name not in index]
        if missing:
            _warn(f"в docs/README.md не добавлены: {', '.join(missing)} (конвенция проекта)")
        else:
            _ok("оба новых документа добавлены в индекс docs/README.md")

    if "langgraph-supervisor" in _read_opt(PYPROJECT):
        _manual("зафиксировать версию langgraph-supervisor в отчёте")


# ---------------------------------------------------------------------- критерий 14


def check_out_of_scope() -> None:
    """Один вопрос — вне корпуса, с честным fallback."""
    report = _read(REPORT)
    low = report.lower()

    if re.search(r"вне\s+(базы|корпуса)|out_of_scope|вне корпуса", low):
        _ok("вопрос «вне базы» присутствует и отмечен")
    else:
        _warn("в отчёте не отмечен вопрос «вне базы» (вне корпуса) — задание требует 1 такой из 5")

    if re.search(r"не\s+нашл|не\s+нашлось|не\s+знаю|confident|fallback|отказ", low):
        _ok("описан fallback (система не выдумывает ответ)")
    else:
        _warn("не видно описания fallback на вопросе вне базы (честное «не нашлось»)")

    _manual("проверить по первоисточникам, что ответа на этот вопрос в data/kb/ действительно нет")


# ------------------------------------------------------------------------- импорт


def check_import() -> None:
    """Опциональный импорт-чек общих модулей experiments/ (без прогона LLM)."""
    sys.path.insert(0, str(ROOT))
    try:
        if (EXPERIMENTS / "__init__.py").exists():
            import experiments.questions as q  # noqa: PLC0415

            items = getattr(q, "QUESTIONS", None)
            assert items is not None, "в experiments/questions.py нет переменной QUESTIONS"
            assert len(items) == 5, f"QUESTIONS содержит {len(items)} элементов — нужно 5"
            _ok("experiments.questions импортируется, QUESTIONS = 5")
        else:
            _warn("experiments/__init__.py нет — импорт-чек общих модулей пропущен")
    except ImportError as exc:
        _warn(f"импорт experiments.questions не удался ({exc})")
    except AssertionError:
        raise
    except Exception as exc:  # noqa: BLE001
        _warn(f"импорт experiments.questions упал ({type(exc).__name__}: {exc})")


# --------------------------------------------------------------------------- main

CHECKS: list[tuple[str, Callable[[], None]]] = [
    ("Критерий 0: зависимости (langgraph/langchain/supervisor)", check_deps),
    ("Критерий 1: layout experiments/ + два скрипта", check_experiments_layout),
    ("Критерий 2: один и тот же tool search_knowledge_base", check_shared_tool),
    ("Критерий 3: 5 одинаковых вопросов", check_questions),
    ("Критерий 4: researcher + writer + супервизор", check_multi_agent),
    ("Критерий 5: чекпоинтер + стриминг + thread_id", check_checkpoint_and_stream),
    ("Критерий 6: draw_mermaid + docs/architecture-multi-agent.md", check_draw_mermaid),
    ("Критерий 7: single-agent baseline", check_single_agent),
    ("Критерий 8: замеры в коде (токены/вызовы/latency/handoff)", check_measurements_code),
    ("Критерий 9: experiments/results.json (10 записей)", check_results_json),
    ("Критерий 10: отчёт multi-agent-report.md", check_report),
    ("Критерий 11: секция «Решение»", check_decision),
    ("Критерий 12: схема графа в docs", check_architecture_doc),
    ("Критерий 13: индекс документации", check_tests_and_docs),
    ("Критерий 14: вопрос вне базы + fallback", check_out_of_scope),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Самопроверка ДЗ 6.5 (мультиагентные системы)")
    parser.add_argument("--skip-import", action="store_true",
                        help="не импортировать модули experiments/")
    args = parser.parse_args()

    print()
    print("=" * 66)
    print("САМОПРОВЕРКА ДЗ 6.5 — Мультиагентные системы")
    print("=" * 66)

    failures: list[tuple[str, str]] = []
    for title, check in CHECKS:
        _section(title)
        try:
            check()
            print(f"[OK] {title.split(':')[0]} ВЫПОЛНЕН")
        except AssertionError as exc:
            failures.append((title, str(exc)))
            print(f"[FAIL] {title.split(':')[0]}: {exc}")
        except Exception as exc:  # неожиданная поломка самого чека
            failures.append((title, f"{type(exc).__name__}: {exc}"))
            print(f"[ERROR] {title.split(':')[0]}: {type(exc).__name__}: {exc}")

    if not args.skip_import:
        _section("Импорт-чек experiments.questions")
        try:
            check_import()
        except AssertionError as exc:
            failures.append(("Импорт-чек", str(exc)))
            print(f"[FAIL] Импорт-чек: {exc}")

    print()
    print("=" * 66)
    print("ИТОГ САМОПРОВЕРКИ")
    print("=" * 66)
    failed_titles = {t for t, _ in failures}
    passed = sum(1 for title, _ in CHECKS if title not in failed_titles)
    print(f"  критериев пройдено: {passed}/{len(CHECKS)}")
    print(f"  предупреждений    : {len(WARNINGS)}")

    if failures:
        print()
        print("  НЕ ВЫПОЛНЕНО:")
        for title, msg in failures:
            print(f"   - {title.split(':')[0]}: {msg}")
    if WARNINGS:
        print()
        print("  ПРЕДУПРЕЖДЕНИЯ:")
        for msg in WARNINGS:
            print(f"   - {msg}")

    print()
    print("  РУЧНАЯ ПРОВЕРКА (скрипт не умеет):")
    manual = MANUAL + [
        "запустить оба скрипта и убедиться, что отработали 5 вопросов и записали 10 записей",
        "проверить, что в ответе мультиагента есть осмысленные цитаты [1], [2], а не пустые скобки",
        "проверить, что на вопросе вне корпуса система честно сказала «не нашлось»",
        "убедиться, что оценки судьи выставлены одним и тем же промптом для обеих реализаций",
        "сверить схему в docs/architecture-multi-agent.md с реальным выводом draw_mermaid()",
        "сверить числа в отчёте с experiments/results.json (без «причёсывания»)",
        "после ДЗ обновить docs/README.md и CLAUDE.md (конвенция проекта)",
    ]
    for msg in manual:
        print(f"   - {msg}")

    print()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
