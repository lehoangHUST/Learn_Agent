"""Node ``finalize_response``: chuẩn hoá đầu ra.

Mọi đường đi trong graph - thành công, một phần, thất bại, bị từ chối - đều kết thúc
ở đây. Nhờ vậy người gọi luôn nhận đúng một schema, và ``status`` có nghĩa xác định:

* ``succeeded`` - làm được việc đã dự định, không có lỗi.
* ``partial``   - có kết quả dùng được nhưng cũng có lỗi (ví dụ detect xong, search
                  lỗi vector store). Người gọi phải biết để không đọc thiếu.
* ``failed``    - không có kết quả nào dùng được.

Node này không sinh dữ liệu mới. Nó chỉ quyết định ``status``, bảo đảm ``answer``
không rỗng, và bỏ trùng trong ``warnings``.
"""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from app.agent.nodes.common import graph_node
from app.agent.state import (
    STATUS_FAILED,
    STATUS_PARTIAL,
    STATUS_SUCCEEDED,
    FaceAgentContext,
    FaceAgentState,
)


@graph_node("finalize_response")
def finalize_response(state: FaceAgentState, runtime: Runtime[FaceAgentContext]) -> dict[str, Any]:
    errors = list(state.get("errors") or [])
    has_result = bool(state.get("detection_summary") or state.get("search_summary"))
    answer = (state.get("answer") or "").strip()

    if not errors:
        status = STATUS_SUCCEEDED
    elif has_result:
        # Có lỗi nhưng vẫn có dữ liệu dùng được -> partial.
        status = STATUS_PARTIAL
    else:
        # Lưu ý: một câu trả lời *giải thích lỗi* KHÔNG làm request thành partial.
        # Nếu tính như vậy thì gần như không bao giờ có status failed, vì hệ thống
        # luôn cố giải thích - và người gọi sẽ mất đi cách phân biệt "có dữ liệu
        # nhưng thiếu" với "không có gì cả".
        status = STATUS_FAILED

    if not answer:
        if errors:
            codes = ", ".join(sorted({str(e.get("code")) for e in errors}))
            answer = (
                "Không hoàn thành được yêu cầu. Mã lỗi: " + codes + ". "
                "Xem trường 'errors' để biết chi tiết."
            )
        else:  # pragma: no cover - không có đường nào tới đây
            answer = "Không có dữ liệu để trả lời."

    # Không cần dedup ở đây: channel ``warnings`` dùng reducer ``dedup_append`` nên
    # việc bỏ trùng đã xảy ra ở mỗi lần gộp state.
    warnings = list(state.get("warnings") or [])

    # Ghi tường minh mọi field của output schema, kể cả field chưa node nào chạm tới.
    # LangGraph chỉ đưa ra output những channel ĐÃ có giá trị, nên nếu không làm bước
    # này thì response thiếu key - và client phải viết `.get()` phòng bị ở mọi field.
    # Hợp đồng đầu ra là việc của node finalize, không phải việc của client.
    return {
        "status": status,
        "answer": answer,
        "answer_source": state.get("answer_source") or "template",
        "intent": state.get("intent") or "unsupported_request",
        "intent_source": state.get("intent_source") or "rule",
        "detection_result_id": state.get("detection_result_id"),
        "search_result_id": state.get("search_result_id"),
        "detection_summary": state.get("detection_summary"),
        "search_summary": state.get("search_summary"),
        "analysis": {
            **(state.get("analysis") or {}),
            "final_status": status,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "executed_steps": list(state.get("executed_steps") or []),
        },
    }
