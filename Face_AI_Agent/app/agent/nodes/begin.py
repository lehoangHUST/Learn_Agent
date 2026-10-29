"""Node ``begin_run``: mốc bắt đầu một lượt.

Vì sao cần một node chỉ để dọn state
------------------------------------
Checkpoint của LangGraph sống theo ``thread_id``. Ở lượt thứ hai của cùng một cuộc
hội thoại, state được **nạp lại** từ checkpoint - kể cả những field mô tả lượt
trước: ``executed_steps``, ``errors``, ``intent``, ``detection_summary``. Nếu không
dọn thì lượt 2 báo là nó đã chạy ``detect_faces`` (thật ra là lượt 1 chạy), và lỗi
của lượt 1 vẫn nằm trong response của lượt 3.

Reset không thể gửi từ input vì ``input_schema`` hẹp làm LangGraph lọc bỏ im lặng
mọi key ngoài schema. Và nó cũng không thể nằm trong ``validate_input``: node đó có
thể lỗi, mà nhánh lỗi của decorator chỉ trả về ``errors`` - lúc đó phần reset không
được gửi và state cũ vẫn còn.

Nên việc dọn thuộc về một node riêng, **không thể lỗi**, chạy trước mọi thứ khác.
Node này cố tình *không* dùng decorator ``graph_node``: decorator sẽ nối vào
``trace``/``executed_steps`` trong cùng một update mà node này đang đặt về ``None``,
hai thứ đó xung đột nhau. Đổi lại, ``begin_run`` không tự ghi tên mình vào
``executed_steps`` - nó vẫn hiện ra như một span trong Langfuse.
"""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from app.agent.state import (
    STATUS_RUNNING,
    FaceAgentContext,
    FaceAgentState,
    run_scoped_reset,
)


def begin_run(state: FaceAgentState, runtime: Runtime[FaceAgentContext]) -> dict[str, Any]:
    """Xoá state phạm vi-một-lần-chạy. Không gọi I/O, không thể lỗi."""
    return {**run_scoped_reset(), "status": STATUS_RUNNING}
