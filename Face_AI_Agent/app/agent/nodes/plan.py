"""Node ``plan_execution``: biến intent thành danh sách bước, và chặn sớm việc không làm được.

Kế hoạch ở đây được **tính bằng code**, không hỏi LLM. Lý do: với 6 intent và hai
dữ kiện (có ảnh mới? có kết quả cũ?), toàn bộ không gian quyết định là một bảng
nhỏ và đã biết trước. Để LLM "lập kế hoạch" cho một bảng tra cứu là thêm một điểm
có thể lỗi mà không đổi lấy được gì.

Việc quan trọng thứ hai của node này là **không gọi công cụ không cần thiết**:

* ``general_face_question`` và ``unsupported_request`` -> không chạm vào detector,
  không chạm vào index. Một câu hỏi lý thuyết không được tốn một lời gọi GPU.
* ``result_question`` -> đọc kết quả đã lưu, không chạy lại model.
* ``face_search`` mà không có ảnh mới -> dùng embedding của lượt trước, không bắt
  người dùng gửi lại ảnh.

Và nó phát hiện sớm trường hợp thiếu dữ liệu: yêu cầu tìm kiếm mà không có cả ảnh
mới lẫn kết quả cũ thì báo lỗi ngay tại đây, trước khi bất kỳ tài nguyên nào bị
tiêu tốn. Lỗi đó không retry được, và node ``answer_question`` sẽ giải thích cho
người dùng thiếu đúng cái gì.
"""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from app.agent.nodes.common import graph_node
from app.agent.state import (
    INTENT_DETECTION_AND_SEARCH,
    INTENT_FACE_DETECTION,
    INTENT_FACE_SEARCH,
    INTENT_GENERAL_FACE_QUESTION,
    INTENT_RESULT_QUESTION,
    INTENT_UNSUPPORTED,
    FaceAgentContext,
    FaceAgentState,
    PlanStep,
)
from app.errors import UnsupportedRequest, ValidationFailed

STEP_DETECT = "detect_faces"
STEP_EMBED = "generate_embeddings"
STEP_SEARCH = "search_faces"
STEP_ANALYZE = "analyze_results"
STEP_ANSWER = "answer_question"


def _step(name: str, reason: str) -> PlanStep:
    return {"step": name, "reason": reason}


@graph_node("plan_execution")
def plan_execution(state: FaceAgentState, runtime: Runtime[FaceAgentContext]) -> dict[str, Any]:
    caps = state.get("capabilities") or {}
    intent = state.get("intent") or INTENT_GENERAL_FACE_QUESTION

    has_images = bool(caps.get("has_new_images"))
    prior_detection_id = caps.get("prior_detection_result_id")
    prior_search_id = caps.get("prior_search_result_id")

    target_result_id = state.get("target_result_id")
    plan: list[PlanStep] = []

    if intent == INTENT_UNSUPPORTED:
        # Không chạy gì cả. Trả lời giải thích giới hạn là toàn bộ công việc.
        raise UnsupportedRequest(
            "yêu cầu này nằm ngoài phạm vi hệ thống: hệ thống chỉ phát hiện khuôn mặt, "
            "tạo embedding và tìm ứng viên trong tập đã đăng ký. Hệ thống không suy "
            "đoán tuổi, giới tính, cảm xúc, chủng tộc và không phục vụ việc theo dõi "
            "người.",
            details={"intent_reason": caps.get("intent_reason")},
        )

    if intent == INTENT_GENERAL_FACE_QUESTION:
        plan = [_step(STEP_ANSWER, "câu hỏi kiến thức chung, không cần chạy model")]

    elif intent == INTENT_FACE_DETECTION:
        if not has_images:
            if prior_detection_id:
                target_result_id = target_result_id or prior_detection_id
                plan = [
                    _step(STEP_ANALYZE, "không có ảnh mới, đọc kết quả detection đã lưu"),
                    _step(STEP_ANSWER, "trả lời từ kết quả đã có"),
                ]
            else:
                raise ValidationFailed(
                    "yêu cầu phát hiện khuôn mặt nhưng request không có ảnh nào và "
                    "thread cũng chưa có kết quả detection nào. Hãy gửi kèm ảnh.",
                    details={"missing": ["image"]},
                )
        else:
            plan = [
                _step(STEP_DETECT, "phát hiện khuôn mặt trên ảnh mới"),
                _step(STEP_ANALYZE, "tổng hợp kết quả phát hiện"),
                _step(STEP_ANSWER, "trả lời câu hỏi dựa trên kết quả"),
            ]

    elif intent in (INTENT_FACE_SEARCH, INTENT_DETECTION_AND_SEARCH):
        if has_images:
            plan = [
                _step(STEP_DETECT, "cần box + embedding của ảnh truy vấn"),
                _step(STEP_EMBED, "kiểm tra và chọn embedding dùng cho truy vấn"),
                _step(STEP_SEARCH, "tìm top-k trong collection được phép truy cập"),
                _step(STEP_ANALYZE, "áp ngưỡng và xác định trạng thái so khớp"),
                _step(STEP_ANSWER, "trả lời dựa trên kết quả thật"),
            ]
        elif prior_detection_id:
            target_result_id = target_result_id or prior_detection_id
            plan = [
                _step(
                    STEP_EMBED,
                    "không có ảnh mới: dùng embedding đã lưu của kết quả detection trước",
                ),
                _step(STEP_SEARCH, "tìm top-k cho (các) khuôn mặt được chỉ định"),
                _step(STEP_ANALYZE, "áp ngưỡng và xác định trạng thái so khớp"),
                _step(STEP_ANSWER, "trả lời dựa trên kết quả thật"),
            ]
        else:
            raise ValidationFailed(
                "yêu cầu tìm kiếm khuôn mặt nhưng không có ảnh mới và thread chưa có "
                "kết quả detection nào để lấy embedding. Hãy gửi ảnh truy vấn.",
                details={"missing": ["image_or_prior_detection"]},
            )

    elif intent == INTENT_RESULT_QUESTION:
        if not (prior_detection_id or prior_search_id or target_result_id):
            raise ValidationFailed(
                "câu hỏi nói về kết quả trước đó nhưng thread này chưa có kết quả nào "
                "còn hiệu lực (có thể đã hết hạn lưu trữ). Hãy gửi lại ảnh.",
                details={"missing": ["prior_result"]},
            )
        # Ưu tiên kết quả search vì câu hỏi về similarity/thứ hạng nằm ở đó.
        target_result_id = target_result_id or prior_search_id or prior_detection_id
        plan = [
            _step(STEP_ANALYZE, "đọc kết quả đã lưu, không chạy lại model"),
            _step(STEP_ANSWER, "trả lời từ dữ liệu đã lưu"),
        ]

    else:  # pragma: no cover - intent lạ bị chặn bởi Literal của LLM và rule
        raise UnsupportedRequest(f"intent không nhận ra: {intent!r}", details={"intent": intent})

    requested = state.get("requested_face_indices")
    return {
        "execution_plan": plan,
        "target_result_id": target_result_id,
        "capabilities": {
            **caps,
            "planned_steps": [p["step"] for p in plan],
            "uses_prior_result": bool(target_result_id),
            "requested_face_indices": requested,
        },
    }
