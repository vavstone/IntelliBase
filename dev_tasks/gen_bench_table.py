"""Генератор markdown-таблицы бенчмарка из JSONL (результат scripts/bench_agents.py).

Читает JSONL, где на строку приходится одна пара задача×реализация, и печатает
таблицу для docs/agent-graph-report.md (раздел 5). Порядок реализаций фиксирован:
react (Б6.2) -> custom -> prebuilt.

Использование:
    uv run python dev_tasks/gen_bench_table.py --in dev_tasks/bench-6-3.jsonl
    uv run python dev_tasks/gen_bench_table.py --in dev_tasks/bench-6-3.jsonl --out docs/_table.md
"""

import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

IMPL_LABEL = {
    "react": "react (Б6.2)",
    "custom": "custom",
    "prebuilt": "prebuilt",
}
IMPL_ORDER = ["react", "custom", "prebuilt"]


def _load(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _short(q: str) -> str:
    return (q[:28] + "…") if len(q) > 28 else q


def render(rows: list[dict]) -> str:
    # группируем по id задачи, внутри — фиксированный порядок реализаций
    by_id: dict[int, list[dict]] = {}
    for r in rows:
        by_id.setdefault(r["id"], []).append(r)

    header = (
        "| # | Задача | Реализация | latency_ms | prompt_tokens | completion_tokens | total_steps |\n"
        "|---|--------|-----------|-----------:|--------------:|------------------:|------------:|"
    )
    lines = [header]
    for task_id in sorted(by_id):
        group = {r["impl"]: r for r in by_id[task_id]}
        question = group[next(iter(group))]["question"]
        for impl in IMPL_ORDER:
            r = group.get(impl)
            if r is None:
                continue
            lines.append(
                f"| {r['id']} | {_short(question)} | {IMPL_LABEL[impl]} | "
                f"{r['latency_ms']:.1f} | {r['prompt']} | {r['completion']} | {r['steps']} |"
            )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Markdown-таблица бенчмарка из JSONL")
    p.add_argument("--in", dest="input", default="dev_tasks/bench-6-3.jsonl")
    p.add_argument("--out", default=None, help="куда сохранить (по умолчанию — stdout)")
    args = p.parse_args()

    path = Path(args.input)
    assert path.exists(), f"не найден {path} — сначала прогоните scripts/bench_agents.py"
    rows = _load(path)
    assert rows, f"{path} пуст"

    table = render(rows)
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")
        print(f"таблица сохранена в {args.out} ({len(rows)} строк данных)")
    else:
        print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
