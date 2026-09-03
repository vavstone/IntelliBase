"""Сохраняет Mermaid-схему кастомного графа агента в `docs/agent-graph-custom.mmd`.

Структура графа не зависит от реального API-ключа: модель строится офлайн,
сеть не дёргается. Файл открывается в mermaid.live.

    uv run python -m scripts.visualize_graph
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.agent_graph import  custom_graph, prebuilt_graph

def main() -> None:

    mermaid_custom = custom_graph.get_graph().draw_mermaid()
    out = Path("docs/agent-graph-custom.mmd")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(mermaid_custom, encoding="utf-8")
    print(f"saved {out}")


    mermaid_prebuilt = prebuilt_graph.get_graph().draw_mermaid()
    out = Path("docs/agent-graph-prebuilt.mmd")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(mermaid_prebuilt, encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
