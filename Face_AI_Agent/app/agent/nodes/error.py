"""Node ``handle_error``: phân loại lỗi và quyết định retry.

Quyết định dựa trên **``retryable`` của chính exception**, không phải đoán từ nội
dung message hay từ HTTP status. Đây là lý do ``app/errors.py`` đặt ``retryable``
thành thuộc tính của từng lớp lỗi: ví dụ backend trả HTTP 500 vì ảnh không giải mã
được là lỗi 5xx nhưng **không** retry được, trong khi timeout là retry được. Nếu
suy ra từ status code thì cả hai đều thành "5xx, thử lại đi" - và ta sẽ gửi lại
cùng một ảnh lỗi ba lần.

Ba nguyên tắc:

1. **Không retry vô hạn.** Chặn bởi ``graph.max_node_retries`` và thêm một lớp nữa
   là ``recursion_limit`` của LangGraph.
2. **Không retry lỗi đầu vào.** Gọi lại với cùng payload sai thì vẫn sai.
3. **Lỗi không retry được vẫn phải có câu trả lời.** Hệ thống đi tiếp sang
   ``answer_question`` để giải thích thiếu gì, chứ không trả về một response rỗng.
   Ngoại lệ duy nhất: chính ``answer_question`` lỗi - lúc đó đi thẳng tới finalize
   để không thành vòng lặp.

Những lỗi nào retry ở đâu
-------------------------
* Timeout / 5xx của backend vision -> retry ``detect_faces``.
* Lỗi tạm của vector store -> retry ``search_faces``.
* Lỗi LLM -> **không** tới đây: node answer tự rơi về template, vì lúc đó ta đã có
  đủ dữ liệu, chỉ thiếu phần diễn đạt.
* Lỗi đầu vào / phân quyền / unsupported -> không retry, đi giải thích.

LangGraph còn có ``add_node(..., retry_policy=RetryPolicy(...))``, nhưng project này
**không** dùng: ``retry_policy`` chỉ kích hoạt khi exception thoát ra khỏi node, còn
decorator ``graph_node`` cố tình bắt hết để người gọi luôn nhận response đúng schema.
Khai báo cả hai sẽ để lại một cơ chế chết. Chi tiết trong docstring của
``app/agent/graph.py``. Retry thật nằm ở hai tầng: transport (trong adapter vision)
và luồng (node này).
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.runtime import Runtime

from app.agent.nodes.common import graph_node
from app.agent.state import (
    STATUS_FAILED,
    STATUS_PARTIAL,
    STATUS_RUNNING,
    FaceAgentContext,
    FaceAgentState,
)

logger = logging.getLogger(__name__)

# Node nào retry được, và retry bằng cách chạy lại chính nó.
RETRYABLE_NODES = {"detect_faces", "generate_embeddings", "search_faces"}


@graph_node("handle_error")
def handle_error(state: FaceAgentState, runtime: Runtime[FaceAgentContext]) -> dict[str, Any]:
    errors = list(state.get("errors") or [])
    if not errors:  # pragma: no cover - routing chỉ vào đây khi có lỗi
        return {"retry_target": None, "status": STATUS_RUNNING}

    last = errors[-1]
    max_retries = runtime.context.deps.settings.graph.max_node_retries
    retry_count = int(state.get("retry_count") or 0)
    failed_node = str(last.get("node") or "")

    can_retry = (
        bool(last.get("retryable"))
        and failed_node in RETRYABLE_NODES
        and retry_count < max_retries
    )

    if can_retry:
        logger.info(
            "error_retrying",
            extra={
                "request_id": state.get("request_id"),
                "node_name": failed_node,
                "error_code": last.get("code"),
                "attempt": retry_count + 1,
                "max_attempts": max_retries,
            },
        )
        return {
            "retry_count": retry_count + 1,
            "retry_target": failed_node,
            "status": STATUS_RUNNING,
            "warnings": [
                f"node {failed_node} lỗi ({last.get('code')}), đang thử lại lần "
                f"{retry_count + 1}/{max_retries}"
            ],
        }

    # Hết đường retry. Còn dữ liệu nào dùng được không?
    has_partial = bool(state.get("detection_summary") or state.get("search_summary"))
    status = STATUS_PARTIAL if has_partial else STATUS_FAILED

    reason = (
        "lỗi không retry được"
        if not last.get("retryable")
        else f"đã retry {retry_count}/{max_retries} lần nhưng vẫn lỗi"
    )
    logger.warning(
        "error_final",
        extra={
            "request_id": state.get("request_id"),
            "node_name": failed_node,
            "error_code": last.get("code"),
            "status": status,
            "reason": reason,
        },
    )

    return {
        "retry_target": None,
        "status": status,
        "warnings": [f"{failed_node}: {last.get('code')} - {reason}"],
    }
