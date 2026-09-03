"""CLI-скрипт для проверки критериев самопроверки ДЗ 6.3 (LangGraph: основы).

Критерии:
0. Зависимости langgraph>=1.0,<2 / langchain>=1.0 / langchain-openai / langchain-core
   объявлены в pyproject.toml; установленная версия langgraph — 1.x.
1. AgentState: messages c add_messages, iteration_count: int, tool_results с operator.add.
   В state нет SDK-клиентов, сессий и API-ключей.
2. Инструменты оформлены через @tool с содержательным docstring (2-3 штуки, один доменный).
3. Три узла-корутины: call_model, execute_tool, force_finish. execute_tool не падает
   на неизвестном инструменте и заполняет ToolMessage(tool_call_id=...).
4. Router — синхронная чистая функция с аннотацией Literal[...], возвращает имена веток.
5. Сборка custom_graph: StateGraph + 3 add_node + START/END + add_conditional_edges +
   обратное ребро execute_tool -> call_model + compile().
6. prebuilt_graph через langchain.agents.create_agent (или create_react_agent + TODO).
7. Явный стоп-кран: iteration_count >= MAX_ITERATIONS уводит в force_finish.
8. scripts/visualize_graph.py + docs/agent-graph-custom.mmd + docs/agent-graph-prebuilt.mmd.
9. scripts/bench_agents.py: 5 задач x 3 реализации x >=3 прогона, perf_counter, токены,
   thread_id в config.
10. Отчёт docs/agent-graph-report.md: 8 разделов (конфигурация, state contract, router,
    mermaid, таблица бенчмарка, custom vs prebuilt, баг отладки, блокеры персистентности).

Скрипт не останавливается на первой ошибке: проходит все критерии и печатает итог.

Использование:
    uv run python dev_tasks/verify_6_3.py
    uv run python dev_tasks/verify_6_3.py --skip-import   # без импорта agent_graph (не нужен Qdrant)
"""

import argparse
import ast
import re
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
AGENT_GRAPH = ROOT / "app" / "services" / "agent_graph.py"
GRAPH_TOOLS = ROOT / "app" / "tools" / "graph_tools.py"
VISUALIZE = ROOT / "scripts" / "visualize_graph.py"
BENCH = ROOT / "scripts" / "bench_agents.py"
MMD_CUSTOM = ROOT / "docs" / "agent-graph-custom.mmd"
MMD_PREBUILT = ROOT / "docs" / "agent-graph-prebuilt.mmd"
REPORT = ROOT / "docs" / "agent-graph-report.md"

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


def _parse(path: Path) -> tuple[str, ast.Module]:
    src = _read(path)
    try:
        return src, ast.parse(src)
    except SyntaxError as exc:  # noqa: TRY302 — нужен внятный текст в отчёте
        raise AssertionError(f"{path.name}: синтаксическая ошибка — {exc}") from exc


def _funcs(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        n.name: n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
    return names


def _tools_source() -> tuple[Path, str, ast.Module]:
    """Инструменты ищем в app/tools/graph_tools.py, иначе в agent_graph.py."""
    path = GRAPH_TOOLS if GRAPH_TOOLS.exists() else AGENT_GRAPH
    src, tree = _parse(path)
    return path, src, tree


# ----------------------------------------------------------------------- критерий 0


def check_deps() -> None:
    """Зависимости LangGraph/LangChain 1.x объявлены и установлены."""
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

    langgraph_spec = spec_for("langgraph")
    langchain_spec = spec_for("langchain")
    assert langgraph_spec, "в pyproject.toml нет зависимости langgraph (uv add 'langgraph>=1.0,<2')"
    assert langchain_spec, "в pyproject.toml нет зависимости langchain (uv add 'langchain>=1.0,<2')"
    _ok(f"объявлены: {langgraph_spec}; {langchain_spec}")

    for name, dep in (("langgraph", langgraph_spec), ("langchain", langchain_spec)):
        if not re.search(r">=\s*1(\.|,|\s|$)", dep):
            _warn(f"{name}: в спецификации '{dep}' не видно нижней границы >=1.0")

    provider = spec_for("langchain-openai") or spec_for("langchain-anthropic")
    assert provider, "нет провайдер-пакета: langchain-openai или langchain-anthropic"
    _ok(f"провайдер-пакет: {provider}")

    if not spec_for("langchain-core"):
        _warn("langchain-core не объявлен явно (задание просит зафиксировать и его)")

    # фактически установленная версия (langgraph 1.x не имеет атрибута __version__ —
    # версию берём из importlib.metadata)
    try:
        import importlib.metadata as _md  # noqa: PLC0415

        version = _md.version("langgraph")
        assert str(version).startswith("1."), f"установлен langgraph {version}, нужен 1.x"
        _ok(f"установлен langgraph {version}")
    except _md.PackageNotFoundError:
        _warn("langgraph не установлен в текущем окружении — выполните `uv sync`")


# ----------------------------------------------------------------------- критерий 1

_FORBIDDEN_FIELD_PARTS = ("client", "api_key", "apikey", "secret", "session", "connection")
_FORBIDDEN_ANNOTATIONS = ("OpenAI", "ChatOpenAI", "AsyncOpenAI", "httpx", "Anthropic", "SecretStr")


def check_state() -> None:
    """AgentState: поля, reducers, отсутствие клиентов/ключей."""
    src, tree = _parse(AGENT_GRAPH)

    state_cls = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "AgentState"),
        None,
    )
    assert state_cls is not None, "в agent_graph.py нет класса AgentState"

    bases = {ast.unparse(b) for b in state_cls.bases}
    assert bases & {"TypedDict", "MessagesState"}, (
        f"AgentState должен наследоваться от TypedDict или MessagesState, а не от {bases or '{}'}"
    )
    inherits_messages_state = "MessagesState" in bases
    _ok(f"AgentState({', '.join(bases)})")

    fields: dict[str, str] = {}
    for stmt in state_cls.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            fields[stmt.target.id] = ast.unparse(stmt.annotation)

    # messages + add_messages
    if "messages" in fields:
        assert "add_messages" in fields["messages"], (
            "поле messages объявлено без reducer add_messages — история будет перезаписываться"
        )
        assert "Annotated" in fields["messages"], "messages должен быть Annotated[...]"
        _ok("messages: Annotated[..., add_messages]")
    else:
        assert inherits_messages_state, "в AgentState нет поля messages"
        _ok("messages наследуется из MessagesState (add_messages внутри)")

    # iteration_count — без Annotated (reducer replace)
    assert "iteration_count" in fields, "в AgentState нет поля iteration_count"
    assert "int" in fields["iteration_count"], "iteration_count должен быть int"
    if "Annotated" in fields["iteration_count"]:
        _warn("iteration_count объявлен через Annotated — задание просит reducer по умолчанию (replace)")
    else:
        _ok("iteration_count: int (reducer replace)")

    # tool_results — operator.add
    assert "tool_results" in fields, "в AgentState нет поля tool_results"
    ann = fields["tool_results"]
    assert "Annotated" in ann and re.search(r"operator\.add|\badd\b|iadd", ann), (
        f"tool_results должен быть Annotated[list[dict], operator.add], сейчас: {ann}"
    )
    _ok("tool_results: Annotated[..., operator.add]")

    # запрет клиентов/ключей в state
    for name, annotation in fields.items():
        low = name.lower()
        bad_name = next((p for p in _FORBIDDEN_FIELD_PARTS if p in low), None)
        assert bad_name is None, f"в state лежит несериализуемое/секретное поле '{name}'"
        bad_ann = next((a for a in _FORBIDDEN_ANNOTATIONS if a in annotation), None)
        assert bad_ann is None, f"в state лежит поле '{name}: {annotation}' (клиент/секрет)"
    _ok("в state нет клиентов, сессий и API-ключей")

    if "add_messages" not in src:
        _warn("в файле нет импорта add_messages из langgraph.graph.message")


# ----------------------------------------------------------------------- критерий 2


def check_tools() -> None:
    """Инструменты через @tool с осмысленным docstring."""
    path, src, tree = _tools_source()
    print(f"  источник инструментов: {path.relative_to(ROOT)}")

    tools = [
        fn
        for fn in _funcs(tree).values()
        if "tool" in _decorator_names(fn) or "StructuredTool" in _decorator_names(fn)
    ]
    assert len(tools) >= 2, (
        f"найдено @tool-функций: {len(tools)} — задание требует 2-3 (включая доменный)"
    )
    if len(tools) > 4:
        _warn(f"инструментов {len(tools)} — задание просит 2-3, лишние размывают выбор модели")

    for fn in tools:
        doc = ast.get_docstring(fn)
        assert doc, f"у инструмента {fn.name} нет docstring — LLM не увидит description"
        if len(doc.strip()) < 40:
            _warn(f"docstring у {fn.name} короче 40 символов — опишите когда вызывать и что возвращает")
    _ok(f"@tool-инструментов: {len(tools)} ({', '.join(t.name for t in tools)}), у всех есть docstring")

    domain = [t.name for t in tools if re.search(r"knowledge|search|rag|kb|поиск", t.name, re.I)]
    if domain:
        _ok(f"доменный инструмент: {', '.join(domain)}")
    else:
        _warn("не видно доменного инструмента (search_knowledge_base) — задание просит хотя бы один")

    assert re.search(r"^\s*TOOLS\s*=", src, re.M), "нет списка TOOLS = [...] для передачи в оба графа"
    _ok("список TOOLS определён")

    for fn in tools:
        body = ast.get_source_segment(src, fn) or ""
        if "asyncio.run(" in body:
            _warn(
                f"{fn.name} вызывает asyncio.run() — внутри async-узла графа это падение;"
                " сделайте инструмент async и используйте await"
            )


# ----------------------------------------------------------------------- критерий 3


def check_nodes() -> None:
    """Три узла-корутины и корректный execute_tool."""
    src, tree = _parse(AGENT_GRAPH)
    funcs = _funcs(tree)

    for name in ("call_model", "execute_tool", "force_finish"):
        assert name in funcs, f"нет узла {name}()"
        assert isinstance(funcs[name], ast.AsyncFunctionDef), f"узел {name} должен быть async def"
    _ok("узлы call_model / execute_tool / force_finish объявлены как async def")

    call_model = ast.get_source_segment(src, funcs["call_model"]) or ""
    assert "ainvoke" in call_model, "call_model должен вызывать await model...ainvoke(...)"
    assert "iteration_count" in call_model, "call_model должен инкрементировать iteration_count"
    assert "messages" in call_model, "call_model должен возвращать {'messages': [response], ...}"
    _ok("call_model: ainvoke + инкремент iteration_count + возврат messages")

    if "bind_tools" not in src:
        _warn("в файле нет bind_tools — модель не увидит инструменты")

    exec_tool = ast.get_source_segment(src, funcs["execute_tool"]) or ""
    assert "tool_calls" in exec_tool, "execute_tool должен читать last_message.tool_calls"
    assert "ToolMessage" in exec_tool, "execute_tool должен оборачивать результат в ToolMessage"
    assert "tool_call_id" in exec_tool, "в ToolMessage обязателен tool_call_id"
    assert re.search(r"not in|\.get\(|try:|except", exec_tool), (
        "execute_tool не обрабатывает неизвестный инструмент — граф упадёт вместо ToolMessage с ошибкой"
    )
    assert "tool_results" in exec_tool, "execute_tool должен пополнять tool_results"
    _ok("execute_tool: tool_calls + ToolMessage(tool_call_id) + обработка неизвестного tool + tool_results")

    finish = ast.get_source_segment(src, funcs["force_finish"]) or ""
    if not re.search(r"AIMessage|messages|answer|лимит", finish, re.I):
        _warn(
            "force_finish ничего не формирует — при исчерпании лимита прогон закончится без текста;"
            " добавьте AIMessage с сообщением о лимите"
        )
    else:
        _ok("force_finish формирует финальное состояние")


# ----------------------------------------------------------------------- критерий 4


def check_router() -> None:
    """Router — синхронная чистая функция с Literal-аннотацией."""
    src, tree = _parse(AGENT_GRAPH)
    funcs = _funcs(tree)

    router_name = next((n for n in funcs if n.startswith("route")), None)
    assert router_name, "нет router-функции (ожидается route_after_model)"
    router = funcs[router_name]
    if router_name != "route_after_model":
        _warn(f"router называется {router_name}, задание предлагает route_after_model")

    assert not isinstance(router, ast.AsyncFunctionDef), "router должен быть синхронной функцией"

    assert router.returns is not None, "у router нет аннотации возвращаемого типа"
    returns = ast.unparse(router.returns)
    assert "Literal" in returns, f"аннотация возврата router должна быть Literal[...], сейчас {returns}"
    _ok(f"{router_name}(state) -> {returns}")

    # чистота: нет await, нет сетевых вызовов, нет записи в state
    assert not any(isinstance(n, ast.Await) for n in ast.walk(router)), "в router есть await — он должен быть чистым"
    body = ast.get_source_segment(src, router) or ""
    dirty = [kw for kw in ("ainvoke", "invoke(", "create(", "requests.", "httpx.", "print(") if kw in body]
    assert not dirty, f"router делает сайд-эффекты: {dirty}"

    for node in ast.walk(router):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                assert not isinstance(target, ast.Subscript), "router пишет в state — он должен только читать"
    _ok("router не пишет в state, не делает сетевых вызовов и await")

    # возвращает строки-имена веток
    returned = {
        n.value.value
        for n in ast.walk(router)
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Constant)
    }
    assert returned, "router не возвращает строковых констант — вернуть нужно имя ветки"
    assert all(isinstance(v, str) for v in returned), (
        f"router возвращает не строки ({returned}) — типичный баг: возврат bool вместо имени ветки"
    )
    assert returned <= {"execute_tool", "force_finish"}, (
        f"router возвращает {returned}, ожидались только 'execute_tool' / 'force_finish'"
    )
    _ok(f"router возвращает имена веток: {sorted(returned)}")


# ----------------------------------------------------------------------- критерий 5


def check_custom_graph() -> None:
    """Сборка кастомного StateGraph."""
    src = _read(AGENT_GRAPH)

    assert "StateGraph(" in src, "нет StateGraph(AgentState)"
    for node in ("call_model", "execute_tool", "force_finish"):
        assert re.search(rf"add_node\(\s*[\"']{node}[\"']", src), f"нет add_node('{node}', ...)"
    _ok("StateGraph + три add_node")

    assert re.search(r"add_edge\(\s*START\s*,\s*[\"']call_model[\"']", src), "нет add_edge(START, 'call_model')"
    assert re.search(r"add_conditional_edges\(\s*[\"']call_model[\"']", src), (
        "нет add_conditional_edges('call_model', router, {...})"
    )
    assert re.search(r"add_edge\(\s*[\"']execute_tool[\"']\s*,\s*[\"']call_model[\"']", src), (
        "нет обратного ребра execute_tool -> call_model (без него нет цикла ReAct)"
    )
    assert re.search(r"add_edge\(\s*[\"']force_finish[\"']\s*,\s*END", src), "нет add_edge('force_finish', END)"
    _ok("рёбра: START->call_model, conditional, execute_tool->call_model, force_finish->END")

    assert ".compile()" in src or ".compile(" in src, "граф не скомпилирован (.compile())"
    assert re.search(r"^\s*custom_graph\s*=", src, re.M), "нет переменной custom_graph на уровне модуля"
    _ok("custom_graph = builder.compile()")


# ----------------------------------------------------------------------- критерий 6


def check_prebuilt() -> None:
    """prebuilt_graph через create_agent."""
    src = _read(AGENT_GRAPH)

    assert re.search(r"^\s*prebuilt_graph\s*=", src, re.M), "нет переменной prebuilt_graph"

    modern = "from langchain.agents import create_agent" in src or re.search(
        r"from\s+langchain\.agents\s+import\s+\(?[^)\n]*create_agent", src
    )
    legacy = "create_react_agent" in src

    if modern:
        _ok("используется langchain.agents.create_agent (рекомендуемый путь LangChain 1.x)")
    else:
        assert legacy, "prebuilt-агент собран не через create_agent / create_react_agent"
        assert re.search(r"#.*TODO.*create_agent", src, re.I), (
            "используется deprecated create_react_agent без комментария # TODO: мигрировать на create_agent"
        )
        _warn("prebuilt собран на deprecated create_react_agent (TODO-комментарий на месте)")

    assert re.search(r"tools\s*=\s*(TOOLS|tools)\b", src), (
        "в prebuilt переданы не те же TOOLS"
    )
    assert re.search(r"system_prompt\s*=|prompt\s*=", src), "в prebuilt не передан system_prompt"
    _ok("prebuilt получает те же TOOLS и system_prompt")

    if re.search(r"create_agent\(\s*\n?\s*model\s*=\s*[\"']", src):
        _warn(
            "model передан строкой ('provider:model') — в проекте провайдер DeepSeek,"
            " передавайте экземпляр ChatOpenAI(base_url=...)"
        )


# ----------------------------------------------------------------------- критерий 7


def check_stop_condition() -> None:
    """Явный стоп-кран по iteration_count."""
    src = _read(AGENT_GRAPH)

    consts = re.findall(r"^\s*MAX_ITERATIONS\s*(?::\s*int\s*)?=\s*(\d+)", src, re.M)
    assert consts, "нет константы MAX_ITERATIONS"
    value = int(consts[0])
    assert 3 <= value <= 12, f"MAX_ITERATIONS={value} вне разумного диапазона 3-12 (в задании 6)"
    if value != 6:
        _warn(f"MAX_ITERATIONS={value}, в задании указано 6 — обоснуйте в отчёте")
    if value > 12:
        _warn("при большом MAX_ITERATIONS граф упрётся в recursion_limit=25 раньше force_finish")

    assert re.search(r"iteration_count[^\n]*>=\s*(MAX_ITERATIONS|\d+)", src), (
        "нет проверки iteration_count >= MAX_ITERATIONS"
    )
    _ok(f"стоп-кран: iteration_count >= MAX_ITERATIONS ({value}) -> force_finish")
    _manual("прогнать граф с заведомо бесполезным/сломанным инструментом и убедиться, "
            "что упирается в force_finish, а не крутится до recursion_limit")


# ----------------------------------------------------------------------- критерий 8


def check_visualization() -> None:
    """visualize_graph.py и два .mmd файла."""
    src = _read(VISUALIZE)

    assert "draw_mermaid" in src, "скрипт не вызывает get_graph().draw_mermaid()"
    assert "agent_graph" in src, "скрипт не импортирует графы из app.services.agent_graph"
    assert "custom_graph" in src and "prebuilt_graph" in src, "скрипт визуализирует не оба графа"
    assert "agent-graph-custom.mmd" in src, "скрипт не пишет docs/agent-graph-custom.mmd"
    assert "agent-graph-prebuilt.mmd" in src, "скрипт не пишет docs/agent-graph-prebuilt.mmd"
    _ok("visualize_graph.py: импорт обоих графов + draw_mermaid + два .mmd")

    if "draw_mermaid_png" in src:
        assert re.search(r"try:|except", src), (
            "draw_mermaid_png() без try/except — упадёт без доступа к mermaid.ink/playwright"
        )
        _ok("PNG-вариант обёрнут в try/except (опциональный бонус)")

    for path in (MMD_CUSTOM, MMD_PREBUILT):
        content = _read(path)
        assert len(content.strip()) > 50, f"{path.name} подозрительно пустой"
        assert re.search(r"graph\s+(TD|LR)|flowchart|stateDiagram", content), (
            f"{path.name} не похож на mermaid-схему"
        )
    _ok("docs/agent-graph-custom.mmd и docs/agent-graph-prebuilt.mmd на месте")

    custom = _read(MMD_CUSTOM)
    missing = [n for n in ("call_model", "execute_tool", "force_finish") if n not in custom]
    assert not missing, f"в agent-graph-custom.mmd нет узлов: {missing} — схема не от того графа"
    _ok("в схеме кастомного графа видны все три узла")
    _manual("открыть оба .mmd на mermaid.live и сверить с кодом сборки")


# ----------------------------------------------------------------------- критерий 9


def check_bench() -> None:
    """bench_agents.py: 5 задач x 3 реализации x >=3 прогона."""
    src = _read(BENCH)

    assert "perf_counter" in src, "нет замера latency через time.perf_counter()"
    assert "custom_graph" in src, "бенчмарк не вызывает custom_graph"
    assert "prebuilt_graph" in src, "бенчмарк не вызывает prebuilt_graph"
    assert re.search(r"run_react_with_reflection|agent_react|run_agent|agent_naive", src), (
        "бенчмарк не вызывает baseline из Б6.2 (run_react_with_reflection)"
    )
    _ok("три реализации: baseline Б6.2 + custom_graph + prebuilt_graph")

    numbers = [
        int(v)
        for v in re.findall(r"(?:runs|repeats|n_runs|attempts|повтор\w*)\s*(?::\s*int\s*)?=\s*(\d+)", src, re.I)
    ]
    numbers += [int(v) for v in re.findall(r"--runs[^)]*?default\s*=\s*(\d+)", src, re.S)]
    numbers += [int(v) for v in re.findall(r"range\(\s*(\d+)\s*\)", src)]
    assert numbers, "не видно числа повторов (runs=3 / range(3))"
    assert max(numbers) >= 3, f"повторов меньше трёх: {numbers} — задание требует минимум 3"
    _ok(f"повторов на задачу: {max(numbers)} (>=3)")

    assert re.search(r"tasks-6-2\.json|tasks\.json|TASKS\s*=", src), (
        "бенчмарк не читает набор задач (переиспользуйте dev_tasks/tasks-6-2.json)"
    )
    _ok("набор задач подключён")

    assert re.search(r"usage_metadata|input_tokens|prompt_tokens|get_openai_callback", src), (
        "нет подсчёта токенов (usage_metadata по AIMessage)"
    )
    _ok("токены считаются")
    if "get_openai_callback" in src and "usage_metadata" not in src:
        _warn("get_openai_callback на DeepSeek вернёт нули — считайте usage_metadata по AIMessage")

    assert "thread_id" in src, "нет config={'configurable': {'thread_id': ...}} (шаг 10, бонус)"
    _ok("thread_id передаётся в config (задел под checkpointer)")

    if "asyncio" not in src:
        _warn("в бенчмарке нет asyncio — async-узлы графа требуют ainvoke")


# ---------------------------------------------------------------------- критерий 10

_REPORT_SECTIONS = [
    ("1. Конфигурация", r"конфигурац|модель|temperature"),
    ("2. State contract", r"state\s*contract|reducer|состояни"),
    ("3. Router / stop conditions", r"router|маршрут|stop\s*condition|стоп"),
    ("5. Таблица бенчмарка", r"latency|мс\b|ms\b"),
    ("6. custom vs prebuilt", r"prebuilt"),
    ("7. Баг отладки", r"баг|bug|ошибк"),
    ("8. Блокеры персистентности", r"checkpoint|чекпойнт|персистент|thread_id"),
]


def check_report() -> None:
    """Отчёт docs/agent-graph-report.md — 8 разделов."""
    src = _read(REPORT)
    low = src.lower()

    for title, pattern in _REPORT_SECTIONS:
        assert re.search(pattern, low), f"в отчёте не найден раздел «{title}» (искали /{pattern}/)"
    _ok("разделы 1,2,3,5,6,7,8 присутствуют")

    # раздел 4 — mermaid-блок
    assert re.search(r"```mermaid", src), "нет блока ```mermaid со схемой кастомного графа (раздел 4)"
    mermaid_block = re.search(r"```mermaid(.+?)```", src, re.S)
    assert mermaid_block and "call_model" in mermaid_block.group(1), (
        "в mermaid-блоке нет узла call_model — вставлено содержимое не того файла"
    )
    _ok("раздел 4: mermaid-схема кастомного графа вставлена")

    # таблица бенчмарка: строки с числами
    data_rows = [
        line for line in src.splitlines()
        if line.strip().startswith("|") and len(re.findall(r"\d", line)) >= 4
    ]
    assert len(data_rows) >= 5, (
        f"в таблице бенчмарка {len(data_rows)} строк с числами — ожидалось минимум 5 (5 задач x 3 реализации)"
    )
    _ok(f"таблица бенчмарка: {len(data_rows)} строк с числами")

    assert re.search(r"токен|token", low), "в таблице нет токенов"
    assert "custom" in low, "в отчёте не упоминается custom-реализация"
    assert re.search(r"6\.2|react|наив", low), "в отчёте нет baseline из Б6.2"
    _ok("в отчёте сопоставлены три реализации, есть токены и latency")

    _manual("проверить, что числа в таблице — из реального прогона, а баг в разделе 7 свой")


# ------------------------------------------------------------------------- импорт


def check_import() -> None:
    """Опциональный импорт-чек модуля (без прогона LLM)."""
    sys.path.insert(0, str(ROOT))
    try:
        from app.services.agent_graph import custom_graph, prebuilt_graph  # noqa: PLC0415

        nodes = set(custom_graph.get_graph().nodes)
        for name in ("call_model", "execute_tool", "force_finish"):
            assert name in nodes, f"в скомпилированном графе нет узла {name}: {sorted(nodes)}"
        _ok(f"custom_graph скомпилирован, узлы: {sorted(nodes)}")
        assert prebuilt_graph is not None, "prebuilt_graph равен None"
        _ok("prebuilt_graph создан")
    except ImportError as exc:
        _warn(f"импорт не удался ({exc}) — установите зависимости (uv sync)")
    except AssertionError:
        raise
    except Exception as exc:
        _warn(
            f"импорт agent_graph упал ({type(exc).__name__}: {exc}) — "
            "вероятно, тяжёлая инициализация на импорте (RAG/Qdrant); сделайте её ленивой"
        )


# --------------------------------------------------------------------------- main

CHECKS: list[tuple[str, Callable[[], None]]] = [
    ("Критерий 0: зависимости langgraph / langchain 1.x", check_deps),
    ("Критерий 1: AgentState (поля, reducers, никаких клиентов)", check_state),
    ("Критерий 2: инструменты через @tool", check_tools),
    ("Критерий 3: узлы call_model / execute_tool / force_finish", check_nodes),
    ("Критерий 4: router — чистая функция с Literal", check_router),
    ("Критерий 5: сборка custom_graph (StateGraph)", check_custom_graph),
    ("Критерий 6: prebuilt_graph через create_agent", check_prebuilt),
    ("Критерий 7: стоп-кран по iteration_count", check_stop_condition),
    ("Критерий 8: визуализация графов (.mmd)", check_visualization),
    ("Критерий 9: бенчмарк 5 задач x 3 реализации x 3 прогона", check_bench),
    ("Критерий 10: отчёт docs/agent-graph-report.md", check_report),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Самопроверка ДЗ 6.3 (LangGraph)")
    parser.add_argument("--skip-import", action="store_true",
                        help="не импортировать app.services.agent_graph (не нужен Qdrant/сеть)")
    args = parser.parse_args()

    print()
    print("=" * 66)
    print("САМОПРОВЕРКА ДЗ 6.3 — LangGraph: основы")
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
        _section("Импорт-чек app.services.agent_graph")
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
        "прогнать 5 задач на трёх реализациях и сверить числа в отчёте",
        "на провокационной задаче ни одна реализация не должна вызывать инструменты",
        "docstring каждого инструмента отвечает на 4 вопроса чек-листа (6.2)",
        "после ДЗ обновить docs/README.md и CLAUDE.md (конвенция проекта)",
    ]
    for msg in manual:
        print(f"   - {msg}")

    print()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
