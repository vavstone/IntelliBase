"""CLI-обёртка: эквивалент `python -m app.services.agent_naive`.

Запуск из корня llm-service: `python scripts/run_naive_agent.py "<задача>" --trace`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.agent_naive import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())