"""In sơ đồ Mermaid của graph, vẽ từ chính object đã compile.

Vì sao đáng có script riêng: sơ đồ vẽ tay trong tài liệu luôn lạc hậu sau vài lần
refactor, còn sơ đồ này không thể lệch với code. Chạy nó sau khi sửa node/edge để
*nhìn* thấy topology thay vì đọc lại các lời gọi ``add_edge``.

Phần lớn bug LangGraph của người mới là bug cấu trúc chứ không phải bug logic:
thiếu một cạnh, nối sai node, một node thành dead-end, hoặc hai node vô tình chạy
song song khi tưởng là tuần tự. Nhìn hình thì thấy ngay.

Cách dùng::

    python scripts/draw_graph.py                 # in Mermaid
    python scripts/draw_graph.py --out graph.mmd # ghi ra file (commit được, diff được)
    python scripts/draw_graph.py --nodes         # chỉ liệt kê node
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.graph import build_graph


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--nodes", action="store_true")
    args = parser.parse_args()

    # Compile không cần checkpointer: ta chỉ quan tâm topology, và compile() là chỗ
    # LangGraph kiểm tra node không tới được / cạnh trỏ vào node không tồn tại.
    compiled = build_graph().compile()
    graph = compiled.get_graph()

    if args.nodes:
        for name in sorted(graph.nodes):
            print(name)
        return 0

    mermaid = graph.draw_mermaid()
    if args.out:
        args.out.write_text(mermaid, encoding="utf-8")
        print(f"Đã ghi {args.out} ({len(graph.nodes)} node)")
    else:
        print(mermaid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
