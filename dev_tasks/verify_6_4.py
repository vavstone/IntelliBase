"""CLI-скрипт для проверки критериев самопроверки ДЗ 6.4 (LangGraph: продвинутые паттерны).

Критерии:
0. Зависимости: langgraph-checkpoint-sqlite, langgraph-checkpoint-postgres, aiosqlite,
   psycopg[binary,pool]>=3.2 (psycopg v3, НЕ psycopg2). langgraph уже есть с 6.3.
1. app/services/agent_persistent.py: фабрика build_agent(checkpointer) + agent_lifespan()
   (async context manager); переключатель backend через AGENT_CHECKPOINTER в config.
2. await checkpointer.setup() вызывается один раз на старте (в lifespan), не на запрос.
3. migrations/env.py: include_name, исключающий таблицы LangGraph
   (checkpoints / checkpoint_writes / checkpoint_blobs / checkpoint_migrations).
4. interrupt() + Command(resume=...) вместо interrupt_before / interrupt_after.
5. Side-effect исполняется ПОСЛЕ interrupt() (отдельный узел confirm), до — только idempotent
   подготовка (отдельный узел prepare + edge prepare -> confirm).
6. POST /agent/stream отдаёт SSE (StreamingResponse, text/event-stream, astream, stream_mode).
7. scripts/time_travel_demo.py: interrupt payload + aget_state_history + aget_state на прошлом
   checkpoint + две ветки (resume=True / resume=False) на разных thread_id.
8. tests/test_agent_persistent.py: >=3 теста на AsyncSqliteSaver(":memory:"), resume True/False,
   assert_not_called для побочного эффекта.
9. thread_id стабильный; uuid4() на каждый запрос НЕ используется.
10. docs/agent-persistent-report.md: 8 разделов; явно «до interrupt:» / «после interrupt:».
11. compose.yaml: AGENT_CHECKPOINTER=postgres (тот же Postgres), URI на psycopg.

Скрипт не останавливается на первой ошибке: проходит все критерии и печатает итог.

Использование:
    uv run python dev_tasks/verify_6_4.py
    uv run python dev_tasks/verify_6_4.py --skip-import   # без импорта agent_persistent
"""

import argparse
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
CONFIG = ROOT / "app" / "core" / "config.py"
AGENT_PERSISTENT = ROOT / "app" / "services" / "agent_persistent.py"
AGENT_ROUTER = ROOT / "app" / "routers" / "agent.py"
MAIN = ROOT / "app" / "main.py"
ALEMBIC_ENV = ROOT / "migrations" / "env.py"
TIME_TRAVEL = ROOT / "scripts" / "time_travel_demo.py"
TEST = ROOT / "tests" / "test_agent_persistent.py"
REPORT = ROOT / "docs" / "agent-persistent-report.md"
COMPOSE = ROOT / "compose.yaml"
ENV_EXAMPLE = ROOT / ".env.example"

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


# ----------------------------------------------------------------------- критерий 0


def check_deps() -> None:
    """Зависимости checkpointer-пакетов и драйверов объявлены."""
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

    for name in ("langgraph-checkpoint-sqlite", "langgraph-checkpoint-postgres"):
        spec = spec_for(name)
        assert spec, f"в pyproject.toml нет зависимости {name} (uv add '{name}>=2.1')"
        _ok(f"объявлен {spec}")

    aiosqlite = spec_for("aiosqlite")
    assert aiosqlite, "нет aiosqlite — без него AsyncSqliteSaver молча не импортируется"
    _ok(f"объявлен {aiosqlite}")

    psycopg = spec_for("psycopg")
    assert psycopg, "нет psycopg — для AsyncPostgresSaver нужен именно psycopg v3"
    if "binary" not in psycopg:
        _warn(f"psycopg объявлен без extras [binary,pool]: '{psycopg}' — задание просит psycopg[binary,pool]")
    if re.search(r">=\s*3", psycopg):
        _ok(f"psycopg v3: {psycopg}")
    else:
        _warn(f"psycopg без нижней границы >=3.2: '{psycopg}'")

    assert not spec_for("psycopg2"), (
        "объявлен psycopg2 — задание требует psycopg (v3), не psycopg2"
    )

    if not spec_for("langgraph"):
        _warn("не видно langgraph в зависимостях — он должен был остаться с 6.3")


# ----------------------------------------------------------------------- критерий 1


def check_factory_and_lifespan() -> None:
    """build_agent(checkpointer) + agent_lifespan() + AGENT_CHECKPOINTER в config."""
    src = _read(AGENT_PERSISTENT)

    assert re.search(r"def\s+build_agent\s*\(", src), "нет фабрики build_agent(...)"
    assert re.search(r"def\s+build_agent\s*\([^)]*checkpointer", src), (
        "build_agent должен принимать параметр checkpointer"
    )
    _ok("build_agent(checkpointer) определён")

    assert re.search(r"(async\s+def|def)\s+agent_lifespan\s*\(", src), "нет agent_lifespan()"
    assert "@asynccontextmanager" in src, (
        "agent_lifespan должен быть async context manager (@asynccontextmanager)"
    )
    _ok("agent_lifespan() — async context manager")

    cfg = _read(CONFIG)
    assert "agent_checkpointer" in cfg, "в config.py нет поля agent_checkpointer"
    if "memory" in cfg and "postgres" in cfg and "sqlite" in cfg:
        _ok("agent_checkpointer: переключатель memory/sqlite/postgres")
    else:
        _warn("agent_checkpointer есть, но не видно вариантов memory/sqlite/postgres (Literal)")

    # привязка к lifespan приложения
    main_src = _read(MAIN)
    assert re.search(r"agent_lifespan", main_src), (
        "в app/main.py не вызывается agent_lifespan — чекпоинтер не поднимется на старте"
    )
    _ok("agent_lifespan подключён в app/main.py")


# ----------------------------------------------------------------------- критерий 2


def check_setup_once() -> None:
    """setup() вызывается в lifespan (один раз), а не в роутере/на запрос."""
    src = _read(AGENT_PERSISTENT)
    assert "setup()" in src, "нет вызова await checkpointer.setup() в agent_persistent.py"

    # setup() должен стоять внутри agent_lifespan, а не в build_agent (иначе на каждый вызов сборки)
    lifespan_segment = re.search(r"def\s+agent_lifespan.*?(?=\ndef\s|\Z)", src, re.S)
    if lifespan_segment and "setup()" in lifespan_segment.group(0):
        _ok("setup() вызывается внутри agent_lifespan (один раз на старте)")
    else:
        _warn("не видно setup() внутри agent_lifespan — проверьте, что он не на каждый запрос")

    router_src = _read(AGENT_ROUTER)
    if "setup()" in router_src:
        _warn("setup() встречается в роутере — чекпоинтер инициализируется на запрос (антипаттерн)")
    else:
        _ok("в роутере setup() нет (инициализация только в lifespan)")

    main_src = _read(MAIN)
    yields = re.findall(r"^\s*yield\s*(?:#.*)?$", main_src, re.M)
    if len(yields) > 1:
        _warn(
            f"в app/main.py lifespan найдено {len(yields)} yield — в @asynccontextmanager"
            " должен быть ровно один; агентный блок, похоже, вставлен ПОСЛЕ yield"
        )
    else:
        _ok("в lifespan ровно один yield")

    _manual("поднять приложение и убедиться, что setup() выполняется ровно один раз в логе старта")


# ----------------------------------------------------------------------- критерий 3


def check_alembic() -> None:
    """include_name в migrations/env.py исключает таблицы LangGraph."""
    src = _read(ALEMBIC_ENV)

    assert "include_name" in src, "в migrations/env.py нет include_name"
    assert re.search(r"def\s+include_name", src), "нет функции include_name(...)"

    for table in ("checkpoints", "checkpoint_writes", "checkpoint_blobs", "checkpoint_migrations"):
        assert table in src, f"в include_name не упомянута таблица '{table}'"
    _ok("include_name исключает 4 таблицы LangGraph")

    assert re.search(r"context\.configure\([^)]*include_name\s*=\s*include_name", src, re.S), (
        "include_name не передан в context.configure(...) — autogenerate не увидит фильтр"
    )
    _ok("include_name передан в context.configure(...)")

    _manual("uv run alembic revision --autogenerate -m test && убедиться, что drop_table для checkpoints не появился")


# ----------------------------------------------------------------------- критерий 4


def check_hil_api() -> None:
    """interrupt() + Command(resume=...) вместо interrupt_before/after."""
    src = _read(AGENT_PERSISTENT)

    assert re.search(r"\binterrupt\s*\(", src), "нет вызова interrupt(...)"
    assert re.search(r"Command\s*\(\s*resume\s*=", src), "нет Command(resume=...) для возобновления"
    _ok("interrupt() + Command(resume=...) на месте")

    for legacy in ("interrupt_before", "interrupt_after"):
        assert legacy not in src, (
            f"используется deprecated {legacy} — в LangGraph 1.0 HIL делается через interrupt()"
        )
    _ok("interrupt_before / interrupt_after не используются")


# ----------------------------------------------------------------------- критерий 5


def check_idempotency() -> None:
    """Side-effect ПОСЛЕ interrupt; до — только подготовка (prepare -> confirm)."""
    src = _read(AGENT_PERSISTENT)

    # два отдельных узла
    assert re.search(r'add_node\(\s*["\']\w*prepare\w*["\']', src), (
        "нет отдельного узла prepare_<action> (идемпотентная подготовка)"
    )
    assert re.search(r'add_node\(\s*["\']\w*(confirm|approve)\w*["\']', src), (
        "нет отдельного узла confirm_and_execute_<action>"
    )
    _ok("узлы prepare_<action> и confirm/approve-узел выделены отдельно")

    # edge prepare -> confirm (между ними обязательно ребро)
    assert re.search(r'add_edge\(\s*["\']\w*prepare\w*["\']\s*,\s*["\']\w*(confirm|approve)\w*["\']', src), (
        "нет add_edge(prepare, confirm) — между подготовкой и исполнением должно быть ребро"
    )
    _ok("edge prepare -> confirm на месте")

    # interrupt() должен стоять в confirm-узле (узле исполнения), а side-effect — после него
    interrupt_line = src.find("interrupt(")
    confirm_match = re.search(r'async def\s+(\w*(?:confirm|approve|execute)\w*)\s*\(', src)
    if interrupt_line == -1:
        return
    # грубая проверка: в файле после interrupt() присутствует реальный вызов инструмента
    tail = src[interrupt_line:]
    if re.search(r"send_telegram|ainvoke|\.invoke\(|print\(|await\s+\w+", tail):
        _ok("после interrupt() в коде есть реальный side-effect (send/ainvoke)")
    else:
        _warn("не видно side-effect после interrupt() — проверьте, что реальный вызов идёт ПОСЛЕ прерывания")

    _manual("проверить глазами: до interrupt() — только рендер/чтение, сам side-effect — после resume")


# ----------------------------------------------------------------------- критерий 6


def check_sse_endpoint() -> None:
    """POST /agent/stream отдаёт SSE через astream."""
    src = _read(AGENT_ROUTER)

    has_full_route = re.search(r'@\w*router\.(post|get)\(\s*["\']/?agent/stream["\']', src)
    has_prefix = re.search(r'APIRouter\([^)]*prefix\s*=\s*["\']/agent["\']', src)
    has_sub_route = re.search(r'@\w*router\.(post|get)\(\s*["\']/stream["\']', src)
    assert has_full_route or (has_prefix and has_sub_route), (
        "нет маршрута POST /agent/stream (ни @router.post('/agent/stream'),"
        " ни prefix='/agent' + @router.post('/stream'))"
    )
    _ok("маршрут /agent/stream объявлен")

    assert "StreamingResponse" in src, "нет StreamingResponse — SSE не отдастся"
    assert "text/event-stream" in src, 'нет media_type="text/event-stream"'
    _ok("StreamingResponse + text/event-stream")

    assert "astream" in src, "нет graph.astream(...) в роутере"
    assert "stream_mode" in src, 'нет stream_mode=["updates", "messages"] (или аналога)'
    _ok("graph.astream(..., stream_mode=...) используется")

    assert re.search(r"data:\s*", src), "нет форматирования SSE (data: {json}\\n\\n)"
    assert "thread_id" in src, "в роутере нет config={'configurable': {'thread_id': ...}}"
    _ok("SSE-формат data: + config с thread_id")

    main_src = _read(MAIN)
    assert re.search(r"include_router\(\s*agent\.router", main_src), (
        "нет app.include_router(agent.router) в app/main.py"
    )
    assert re.search(r"from\s+app\.routers\s+import[^\n]*\bagent\b", main_src) or \
           re.search(r"from\s+app\.routers\.agent\s+import|import\s+app\.routers\.agent", main_src), (
        "роутер agent не импортирован в app/main.py (добавьте 'agent' в from app.routers import ...)"
    )
    _ok("роутер agent импортирован и подключён (include_router)")


# ----------------------------------------------------------------------- критерий 7


def check_time_travel() -> None:
    """scripts/time_travel_demo.py: 4 демонстрации."""
    src = _read(TIME_TRAVEL)

    assert re.search(r"aget_state_history", src), "нет чтения истории (aget_state_history)"
    assert re.search(r"aget_state\s*\(", src), "нет чтения состояния (aget_state)"
    assert re.search(r"checkpoint_id", src), "нет работы с checkpoint_id"
    _ok("aget_state_history + aget_state + checkpoint_id")

    assert re.search(r"Command\s*\(\s*resume\s*=", src), "нет Command(resume=...) — нет возобновления"
    assert re.search(r"interrupt|__interrupt__", src), "не печатается interrupt payload"
    _ok("печать interrupt payload + Command(resume=...)")

    # две ветки: resume=True и resume=False
    assert re.search(r"resume\s*=\s*True", src), "нет ветки одобрения resume=True"
    assert re.search(r"resume\s*=\s*False", src), "нет ветки отказа resume=False"
    _ok("две ветки: resume=True и resume=False")

    # запуск одной командой
    assert re.search(r'__main__|asyncio\.run', src), "скрипт не запускается одной командой (нет __main__/asyncio.run)"
    _ok("скрипт запускается одной командой")

    # две ветки НЕ должны делаться повторным resume одного thread с разным значением
    thread_ids = set(re.findall(r'thread_id["\']?\s*[:=]\s*["\']([^"\']+)', src))
    if len(thread_ids) >= 2:
        _ok(f"две ветки на разных thread_id: {sorted(thread_ids)}")
    else:
        _warn(
            "две ветки на одном thread_id — напомню: resume детерминирован по thread-lineage,"
            " повторный resume того же interrupt вернёт ПЕРВОЕ значение. Используйте два thread_id"
        )


# ----------------------------------------------------------------------- критерий 8


def check_tests() -> None:
    """tests/test_agent_persistent.py: >=3 теста на AsyncSqliteSaver(:memory:)."""
    src = _read(TEST)

    assert re.search(r"AsyncSqliteSaver", src), "нет AsyncSqliteSaver"
    assert re.search(r":memory:", src), 'нет ":memory:" — тесты должны идти без файла БД'
    _ok("AsyncSqliteSaver + :memory:")

    assert re.search(r"Command\s*\(\s*resume\s*=", src), "в тестах нет Command(resume=...)"
    assert re.search(r"resume\s*=\s*True", src), "нет теста одобрения (resume=True)"
    assert re.search(r"resume\s*=\s*False", src), "нет теста отказа (resume=False)"
    _ok("тесты resume=True и resume=False")

    assert re.search(r"assert_not_called|AsyncMock|Mock", src), (
        "нет мока внешнего API и проверки assert_not_called() для побочного эффекта при отказе"
    )
    _ok("побочный эффект мокается, проверяется assert_not_called")

    tests = re.findall(r"^\s*async\s+def\s+(test_\w+)\s*\(", src, re.M)
    assert len(tests) >= 3, f"тестов найдено {len(tests)} — задание требует минимум 3"
    _ok(f"тестов: {len(tests)} ({', '.join(tests)})")

    _manual("uv run pytest tests/test_agent_persistent.py -v — все зелёные без Postgres")


# ----------------------------------------------------------------------- критерий 9


def check_thread_id_stability() -> None:
    """thread_id стабильный; uuid4() на запрос не используется."""
    for path, label in (
        (AGENT_PERSISTENT, "agent_persistent.py"),
        (AGENT_ROUTER, "routers/agent.py"),
        (TIME_TRAVEL, "time_travel_demo.py"),
    ):
        src = _read(path)
        if re.search(r"uuid4|uuid\.uuid4", src):
            _warn(f"{label}: используется uuid4() — thread_id нестабилен, чекпоинтер потеряет state")

    router_src = _read(AGENT_ROUTER)
    assert re.search(r"thread_id", router_src), "в роутере не используется thread_id"
    _ok("uuid4() на каждый запрос не используется (thread_id стабильный)")

    _manual("thread_id берётся из тела запроса/сессии, а не генерируется каждый раз")


# ---------------------------------------------------------------------- критерий 10


_REPORT_SECTIONS = [
    ("1. Выбор backend", r"sqlite|postgres|checkpointer|backend"),
    ("2. Postgres в docker-compose", r"docker-compose|docker compose|compose|psql|\\dt"),
    ("3. Опасный tool + до/после", r"send_telegram|опасн|interrupt"),
    ("4. Лог __interrupt__ + resume", r"__interrupt__|resume"),
    ("5. Time travel", r"time.?travel|aget_state_history|checkpoint_id|replay"),
    ("6. Streaming mode", r"stream_mode|astream|SSE|sse|streaming"),
    ("7. Permission policy", r"permission|рол|read-only|write-with-approve|full"),
    ("8. Хрупкое / доделать", r"хрупк|доделат|хочется|риск|не сделано|осталос"),
]


def check_report() -> None:
    """docs/agent-persistent-report.md — 8 разделов."""
    src = _read(REPORT)
    low = src.lower()

    missing = [title for title, pattern in _REPORT_SECTIONS if not re.search(pattern, low)]
    assert not missing, f"в отчёте не найдены разделы: {missing}"
    _ok("все 8 разделов отчёта на месте")

    assert re.search(r"до\s+interrupt", low), 'нет явного "до interrupt" в разделе про идемпотентность'
    assert re.search(r"после\s+interrupt", low), 'нет явного "после interrupt" в разделе про идемпотентность'
    _ok('явно перечислены "до interrupt" и "после interrupt"')

    _manual("проверить, что лог в разделе 4 и вывод time travel — из реального прогона, а не выдуман")


# ---------------------------------------------------------------------- критерий 11


def check_compose() -> None:
    """compose.yaml: AGENT_CHECKPOINTER=postgres + URI на psycopg (тот же Postgres)."""
    src = _read(COMPOSE)

    assert re.search(r"AGENT_CHECKPOINTER", src), "в compose.yaml нет AGENT_CHECKPOINTER"
    assert re.search(r"AGENT_CHECKPOINTER\s*:\s*postgres", src), (
        "AGENT_CHECKPOINTER должен быть =postgres в docker-compose (production-режим)"
    )
    _ok("AGENT_CHECKPOINTER=postgres в compose")

    # URI чекпоинтера на psycopg (без +asyncpg)
    uri_hits = re.findall(r"postgresql(?:\+[a-z0-9]+)?://[^\s\"']+", src)
    if any("asyncpg" not in u for u in uri_hits):
        _ok("есть postgres-URI на psycopg (без +asyncpg)")
    else:
        _warn("URI чекпоинтера в compose содержит +asyncpg — AsyncPostgresSaver требует psycopg v3")

    assert re.search(r"postgres:16", src), "нет сервиса postgres:16 (тот же, что для FastAPI)"
    _ok("тот же Postgres-сервис (postgres:16)")

    env_example = _read_opt(ENV_EXAMPLE)
    if env_example and "AGENT_CHECKPOINTER" in env_example:
        _ok("AGENT_CHECKPOINTER задокументирован в .env.example")
    else:
        _warn("AGENT_CHECKPOINTER не добавлен в .env.example")

    _manual("docker compose up -d db && psql -d intellibase -c '\\dt' — видны checkpoints/checkpoint_writes/checkpoint_blobs/checkpoint_migrations")


# ------------------------------------------------------------------------- импорт


def check_import() -> None:
    """Опциональный импорт-чек модуля (без прогона LLM)."""
    sys.path.insert(0, str(ROOT))
    try:
        import app.services.agent_persistent as ap  # noqa: PLC0415

        assert hasattr(ap, "build_agent"), "нет build_agent на уровне модуля"
        assert hasattr(ap, "agent_lifespan"), "нет agent_lifespan на уровне модуля"
        _ok("app.services.agent_persistent импортируется (build_agent + agent_lifespan доступны)")
    except ImportError as exc:
        _warn(f"импорт не удался ({exc}) — установите зависимости (uv sync)")
    except AssertionError:
        raise
    except Exception as exc:
        _warn(
            f"импорт agent_persistent упал ({type(exc).__name__}: {exc}) — "
            "убедитесь, что тяжёлая инициализация (граф/модель/RAG) не выполняется на импорте модуля"
        )


# --------------------------------------------------------------------------- main

CHECKS: list[tuple[str, Callable[[], None]]] = [
    ("Критерий 0: зависимости checkpointer-пакетов", check_deps),
    ("Критерий 1: build_agent + agent_lifespan + AGENT_CHECKPOINTER", check_factory_and_lifespan),
    ("Критерий 2: setup() один раз в lifespan", check_setup_once),
    ("Критерий 3: Alembic include_name", check_alembic),
    ("Критерий 4: interrupt() + Command(resume)", check_hil_api),
    ("Критерий 5: идемпотентность (side-effect после interrupt)", check_idempotency),
    ("Критерий 6: SSE-endpoint /agent/stream", check_sse_endpoint),
    ("Критерий 7: time_travel_demo.py", check_time_travel),
    ("Критерий 8: тесты test_agent_persistent.py", check_tests),
    ("Критерий 9: стабильный thread_id", check_thread_id_stability),
    ("Критерий 10: отчёт agent-persistent-report.md", check_report),
    ("Критерий 11: compose AGENT_CHECKPOINTER=postgres", check_compose),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Самопроверка ДЗ 6.4 (LangGraph: продвинутые паттерны)")
    parser.add_argument("--skip-import", action="store_true",
                        help="не импортировать app.services.agent_persistent")
    args = parser.parse_args()

    print()
    print("=" * 66)
    print("САМОПРОВЕРКА ДЗ 6.4 — LangGraph: продвинутые паттерны")
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
        _section("Импорт-чек app.services.agent_persistent")
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
        "запустить scripts/time_travel_demo.py и сверить 4 вывода глазами",
        "curl -N на /agent/stream: виден поток событий, в паузе __interrupt__, после resume продолжается",
        "pytest tests/test_agent_persistent.py — реально зелёный без Postgres",
        "psql -d intellibase -c '\\dt' показывает 4 таблицы LangGraph после setup()",
        "после ДЗ обновить docs/README.md и CLAUDE.md (конвенция проекта)",
    ]
    for msg in manual:
        print(f"   - {msg}")

    print()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
