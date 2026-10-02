"""Smoke-проверка поднятого стека IntelliBase.

Отвечает на вопрос «система жива и отвечает», а не «контейнеры запущены»:
health приложения, готовность зависимостей (Redis проверяется через `/ready`),
наличие проиндексированного корпуса в Qdrant, доступность Phoenix и — по флагу
`--with-rag` — сквозной вопрос к RAG с проверкой цитат.

Запуск:
    make smoke                                    # быстрые проверки
    uv run python scripts/smoke.py --with-rag     # + сквозной вопрос к RAG

Адреса берутся из окружения, затем из `.env`, затем из дефолтов (localhost).
Внутри контейнера окружение уже содержит нужные значения (`QDRANT_URL` и т.п.),
поэтому один и тот же скрипт работает и на хосте, и через `docker compose exec`.

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


def check_containers() -> list[Check]:
    """Статусы контейнеров стека (пропускается, если docker недоступен изнутри)."""
    try:
        out = subprocess.run(
            ["docker", "compose", "ps", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
    except (OSError, subprocess.SubprocessError):
        return [Check("docker compose ps", SKIP, "docker недоступен — проверка пропущена")]
    if out.returncode != 0:
        return [Check("docker compose ps", SKIP, out.stderr.strip()[:120] or "нет вывода")]

    checks: list[Check] = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        service = item.get("Service", "?")
        state, health = item.get("State", "?"), item.get("Health", "")
        if state != "running":
            checks.append(Check(f"контейнер {service}", FAIL, f"состояние: {state}"))
        elif health and health != "healthy":
            checks.append(Check(f"контейнер {service}", FAIL, f"health: {health}"))
        else:
            checks.append(Check(f"контейнер {service}", OK, health or state))
    return checks


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

    app_url = os.environ.get("SMOKE_APP_URL", "http://localhost:8000").rstrip("/")
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333").rstrip("/")
    collection = os.environ.get("RAG_COLLECTION", "rag_demo")
    phoenix_url = os.environ.get("SMOKE_PHOENIX_URL", "http://localhost:6006").rstrip("/")
    phoenix_required = os.environ.get("PHOENIX_ENABLED", "false").lower() in {"1", "true", "yes"}

    checks: list[Check] = []
    checks += check_containers()
    checks += check_app(app_url)
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
