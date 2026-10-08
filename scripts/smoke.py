"""Smoke-проверка поднятого стека IntelliBase.

Отвечает на вопрос «система жива и отвечает», а не «контейнеры запущены»:
health приложения, готовность зависимостей (Redis проверяется через `/ready`),
ответ бота, наличие проиндексированного корпуса в Qdrant, доступность Phoenix
и — по флагу `--with-rag` — сквозной вопрос к RAG с проверкой цитат.

Запуск:
    make smoke                                    # быстрые проверки
    uv run python scripts/smoke.py --with-rag     # + сквозной вопрос к RAG

Адреса берутся из окружения, затем из `.env`, затем из дефолтов (localhost).
Внутри контейнера окружение уже содержит нужные значения (`QDRANT_URL`, `BOT_URL`
и т.п.), поэтому один и тот же скрипт работает и на хосте, и через
`docker compose exec`.

Код возврата: 0 — критичные проверки прошли, 1 — есть провал.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Русский вывод на Windows-консоли (cp1251) без этого падает UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TIMEOUT = 15.0
DEFAULT_QUESTION = "Какие требования предъявляются к документам?"

OK, WARN, FAIL, SKIP = "OK", "WARN", "FAIL", "SKIP"
ICON = {OK: "+", WARN: "!", FAIL: "x", SKIP: "-"}


@dataclass
class Check:
    """Одна проверка: что смотрели, чем закончилось, что делать при провале."""

    name: str
    status: str
    detail: str


def load_env(path: Path) -> None:
    """Подхватывает `.env`, не затирая уже заданное окружение.

    Внутри контейнера переменные приходят из compose — их перетирать нельзя
    (`setdefault`), иначе Qdrant будет искаться на localhost вместо `qdrant`.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        # В .env встречаются inline-комментарии (« # ...») — отрезаем их.
        value = value.split(" #", 1)[0].strip()
        os.environ.setdefault(key.strip(), value)


def http(
    url: str,
    method: str = "GET",
    payload: dict | None = None,
    timeout: float = TIMEOUT,
) -> tuple[int, str]:
    """HTTP-запрос без внешних зависимостей. Код 0 — сеть/соединение не удались."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 — сеть/таймаут/DNS, детали в тексте
        return 0, f"{type(exc).__name__}: {exc}"


def service_states() -> list[tuple[str, str, str]] | None:
    """Статусы контейнеров через `docker compose ps`; None — docker недоступен.

    Изнутри контейнера `docker` нет — там сервисы проверяются по сети (check_bot).
    """
    try:
        out = subprocess.run(
            ["docker", "compose", "ps", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None

    states: list[tuple[str, str, str]] = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        states.append((item.get("Service", "?"), item.get("State", "?"), item.get("Health", "")))
    return states


def check_bot() -> Check:
    """Бот отвечает на своём `/health` (порт 9000).

    Внутри сети compose адрес берётся из `BOT_URL` (`http://bot:9000`). Если так
    достучаться не удалось, а мы на хосте — порт 9000 наружу не проброшен (в
    compose он только `expose`) — смотрим состояние контейнера через
    `docker compose ps`.

    Раньше статусы контейнеров собирал make на хосте и передавал их сюда через
    SMOKE_CONTAINERS. Цель в Makefile требовала POSIX-shell (if/then/awk) и на
    Windows, где make исполняет рецепты через cmd.exe, падала целиком — вместе
    со всеми проверками.
    """
    url = os.environ.get("SMOKE_BOT_URL") or os.environ.get("BOT_URL", "http://127.0.0.1:9000")
    # localhost на Windows+Docker Desktop уходит в IPv6 и «залипает» (см. main).
    url = url.rstrip("/").replace("//localhost", "//127.0.0.1")

    status, body = http(f"{url}/health", timeout=5.0)
    if status == 200:
        return Check("bot /health", OK, url)

    states = service_states()
    if states is None:
        return Check("bot /health", FAIL, f"{url} не отвечает ({body[:80]})")
    for service, state, health in states:
        if service != "bot":
            continue
        if state == "running" and health in ("", "healthy"):
            return Check("bot", OK, f"контейнер {health or state} (порт наружу не проброшен)")
        return Check("bot", FAIL, f"контейнер {state}/{health or 'без healthcheck'}")
    return Check("bot", FAIL, f"{url} не отвечает, и контейнера bot нет в compose")


def check_app(base_url: str) -> list[Check]:
    """Liveness и readiness приложения (readiness покрывает Redis и RAG-индекс)."""
    checks = []
    for path, name in (("/health", "app /health"), ("/ready", "app /ready")):
        status, body = http(f"{base_url}{path}")
        if status == 200:
            checks.append(Check(name, OK, "200"))
        else:
            checks.append(Check(name, FAIL, f"код {status}: {body[:120]}"))
    return checks


def check_qdrant(qdrant_url: str, collection: str) -> Check:
    """Коллекция существует и наполнена — иначе RAG отвечает отказом на всё."""
    status, body = http(f"{qdrant_url}/collections/{collection}")
    if status != 200:
        return Check(
            f"qdrant:{collection}",
            FAIL,
            f"коллекция недоступна (код {status}) — запустите `make ingest`",
        )
    try:
        points = json.loads(body)["result"]["points_count"]
    except (KeyError, json.JSONDecodeError, TypeError):
        return Check(f"qdrant:{collection}", FAIL, f"неожиданный ответ: {body[:120]}")
    if points <= 0:
        return Check(
            f"qdrant:{collection}", FAIL, "0 точек — корпус не проиндексирован (`make ingest`)"
        )
    return Check(f"qdrant:{collection}", OK, f"{points} точек")


def check_phoenix(phoenix_url: str, required: bool) -> Check:
    """Phoenix UI. Без включённого трейсинга это предупреждение, не провал."""
    status, _ = http(phoenix_url)
    if status == 200:
        return Check("phoenix UI", OK, phoenix_url)
    status_word = WARN if not required else FAIL
    return Check("phoenix UI", status_word, f"недоступен ({phoenix_url})")


def check_rag(base_url: str, question: str, timeout: float) -> Check:
    """Сквозной вопрос к RAG: ответ должен прийти с источниками.

    Таймаут отдельный и щедрый: запрос включает retrieval + генерацию LLM
    (на CPU-модели это минуты, на облачной — секунды).
    """
    status, body = http(
        f"{base_url}/rag/query", method="POST", payload={"question": question}, timeout=timeout
    )
    if status != 200:
        return Check("rag query", FAIL, f"код {status}: {body[:160]}")
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return Check("rag query", FAIL, f"не JSON: {body[:120]}")
    sources = data.get("sources") or []
    if not sources:
        return Check("rag query", FAIL, f"ответ без источников (top_score={data.get('top_score')})")
    return Check(
        "rag query",
        OK,
        f"{len(sources)} источн., top_score={data.get('top_score'):.3f}, "
        f"confident={data.get('confident')}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-проверка стека IntelliBase")
    parser.add_argument(
        "--with-rag",
        action="store_true",
        help="добавить сквозной вопрос к RAG (требует LLM и наполненный корпус)",
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="вопрос для --with-rag")
    parser.add_argument(
        "--rag-timeout",
        type=float,
        default=float(os.environ.get("SMOKE_RAG_TIMEOUT", "120")),
        help="таймаут сквозного вопроса к RAG, сек (по умолчанию 120)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    load_env(root / ".env")

    # 127.0.0.1, а не localhost: на Windows с Docker Desktop localhost уходит
    # сначала в IPv6 (::1), где проброс портов идёт через wslrelay и изредка
    # «залипает» — запрос висит до таймаута, хотя сервис жив. Просим IPv4 явно.
    app_url = os.environ.get("SMOKE_APP_URL", "http://127.0.0.1:8000").rstrip("/")
    qdrant_url = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333").rstrip("/")
    collection = os.environ.get("RAG_COLLECTION", "rag_demo")
    phoenix_url = os.environ.get("SMOKE_PHOENIX_URL", "http://127.0.0.1:6006").rstrip("/")
    phoenix_required = os.environ.get("PHOENIX_ENABLED", "false").lower() in {"1", "true", "yes"}

    checks: list[Check] = []
    checks += check_app(app_url)
    checks.append(check_bot())
    checks.append(check_qdrant(qdrant_url, collection))
    checks.append(check_phoenix(phoenix_url, phoenix_required))
    if args.with_rag:
        checks.append(check_rag(app_url, args.question, args.rag_timeout))

    print("=== IntelliBase smoke ===")
    for chk in checks:
        print(f"  [{ICON[chk.status]:>1}] {chk.name:<28} {chk.detail}")

    failed = [c for c in checks if c.status == FAIL]
    warned = [c for c in checks if c.status == WARN]
    print("-" * 60)
    if failed:
        print(f"ПРОВАЛ: {len(failed)} из {len(checks)} проверок не прошли")
        return 1
    print(f"OK: {len(checks)} проверок пройдено" + (f", предупреждений: {len(warned)}" if warned else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
