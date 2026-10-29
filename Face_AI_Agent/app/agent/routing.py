"""Conditional edges.

Ý tưởng: **kế hoạch điều khiển luồng.** ``plan_execution`` đã quyết định danh sách
bước; mỗi router chỉ trả lời một câu hỏi rất hẹp - "node vừa chạy có lỗi không, và
bước kế tiếp trong kế hoạch là gì". Nhờ vậy logic định tuyến không bị nhân bản ở
mỗi cạnh, và việc thêm một intent mới chỉ là thêm một kế hoạch, không phải sửa
năm hàm router.

Tất cả hàm ở đây **thuần**: nhận state, trả về tên node. Không I/O, không thời gian,
không ngẫu nhiên -> test được bằng cách dựng dict state bằng tay.

Phát hiện lỗi: đọc channel ``failed_node`` - do decorator ``graph_node`` đặt tên node
khi nó lỗi và đặt lại ``None`` khi nó chạy xong không lỗi.

Hai cách làm sai mà thiết kế này tránh:

* **Đọc ``status``.** Sau khi ``handle_error`` kết luận không retry được, ``status``
  vẫn là ``failed``/``partial`` suốt phần còn lại của lần chạy, nên mọi node sau đó
  lại bị đẩy về ``handle_error``.
* **Đọc ``errors[-1].node``.** Lỗi cũ vẫn nằm trong ``errors`` sau khi retry thành
  công (đó là lịch sử, người gọi cần thấy), nên ``errors[-1].node`` vẫn trỏ vào node
  vừa chạy *thành công* - và router lại đẩy nó về ``handle_error``. Đúng lỗi này đã
  làm một lần retry thành 3 lần trước khi được phát hiện bằng test.
"""

from __future__ import annotations

from typing import Any

from app.agent.state import FaceAgentState

NODE_BEGIN = "begin_run"
NODE_VALIDATE = "validate_input"
NODE_INTENT = "classify_intent"
NODE_PLAN = "plan_execution"
NODE_DETECT = "detect_faces"
NODE_EMBED = "generate_embeddings"
NODE_SEARCH = "search_faces"
NODE_ANALYZE = "analyze_results"
NODE_ANSWER = "answer_question"
NODE_ERROR = "handle_error"
NODE_FINALIZE = "finalize_response"

# Các node có thể xuất hiện trong execution_plan, theo thứ tự thực thi hợp lệ.
PLAN_NODES = (NODE_DETECT, NODE_EMBED, NODE_SEARCH, NODE_ANALYZE, NODE_ANSWER)


def failed_here(state: FaceAgentState | dict[str, Any], node: str) -> bool:
    """Node ``node`` vừa chạy xong và nó lỗi?

    Chỉ đọc ``failed_node`` - tín hiệu về *bước hiện tại*, không phải lịch sử lỗi.
    """
    return state.get("failed_node") == node


def next_planned_step(
    state: FaceAgentState | dict[str, Any],
    after: str | None = None,
) -> str:
    """Bước kế tiếp trong kế hoạch sau ``after``. Hết kế hoạch -> finalize.

    ``after=None`` nghĩa là "bước đầu tiên".
    """
    plan = [p["step"] for p in (state.get("execution_plan") or [])]
    if not plan:
        return NODE_FINALIZE

    if after is None:
        candidates = plan
    else:
        if after not in plan:
            # Node chạy ngoài kế hoạch (ví dụ answer được gọi từ handle_error).
            return NODE_FINALIZE
        candidates = plan[plan.index(after) + 1 :]

    for step in candidates:
        if step in PLAN_NODES:
            return step
    return NODE_FINALIZE


def _route(state: FaceAgentState | dict[str, Any], current: str) -> str:
    return NODE_ERROR if failed_here(state, current) else next_planned_step(state, current)


# --------------------------------------------------------------------------- #
# Router cho từng cạnh
# --------------------------------------------------------------------------- #
def route_after_validate(state: FaceAgentState) -> str:
    return NODE_ERROR if failed_here(state, NODE_VALIDATE) else NODE_INTENT


def route_after_intent(state: FaceAgentState) -> str:
    return NODE_ERROR if failed_here(state, NODE_INTENT) else NODE_PLAN


def route_after_plan(state: FaceAgentState) -> str:
    if failed_here(state, NODE_PLAN):
        return NODE_ERROR
    return next_planned_step(state, None)


def route_after_detect(state: FaceAgentState) -> str:
    return _route(state, NODE_DETECT)


def route_after_embed(state: FaceAgentState) -> str:
    return _route(state, NODE_EMBED)


def route_after_search(state: FaceAgentState) -> str:
    return _route(state, NODE_SEARCH)


def route_after_analyze(state: FaceAgentState) -> str:
    return _route(state, NODE_ANALYZE)


def route_after_answer(state: FaceAgentState) -> str:
    """Sau khi trả lời là kết thúc, kể cả khi node answer lỗi.

    Không đẩy về ``handle_error``: node đó sẽ lại gửi về ``answer_question`` để giải
    thích lỗi, tức là vòng lặp. ``finalize_response`` biết cách tạo một câu trả lời
    tối thiểu từ danh sách lỗi, nên đường này vẫn cho người gọi một response hợp lệ.
    """
    return NODE_FINALIZE


def route_after_error(state: FaceAgentState) -> str:
    """Retry, hay đi giải thích, hay kết thúc."""
    target = state.get("retry_target")
    if target in PLAN_NODES:
        return str(target)

    # Chưa có câu trả lời và lỗi không phải của chính node answer -> đi giải thích.
    last_failed = (state.get("errors") or [{}])[-1].get("node")
    if not (state.get("answer") or "").strip() and last_failed != NODE_ANSWER:
        return NODE_ANSWER

    return NODE_FINALIZE


# Path map tường minh cho mỗi cạnh điều kiện. Khai báo thay vì để LangGraph suy ra
# giúp ``draw_mermaid`` vẽ đúng các nhánh, và giúp ``compile()`` phát hiện ngay nếu
# một router trả về tên node không tồn tại.
PATHS_AFTER_VALIDATE = [NODE_INTENT, NODE_ERROR]
PATHS_AFTER_INTENT = [NODE_PLAN, NODE_ERROR]
PATHS_AFTER_PLAN = [NODE_DETECT, NODE_EMBED, NODE_ANALYZE, NODE_ANSWER, NODE_ERROR, NODE_FINALIZE]
PATHS_AFTER_DETECT = [NODE_EMBED, NODE_SEARCH, NODE_ANALYZE, NODE_ANSWER, NODE_ERROR, NODE_FINALIZE]
PATHS_AFTER_EMBED = [NODE_SEARCH, NODE_ANALYZE, NODE_ANSWER, NODE_ERROR, NODE_FINALIZE]
PATHS_AFTER_SEARCH = [NODE_ANALYZE, NODE_ANSWER, NODE_ERROR, NODE_FINALIZE]
PATHS_AFTER_ANALYZE = [NODE_ANSWER, NODE_ERROR, NODE_FINALIZE]
PATHS_AFTER_ERROR = [NODE_DETECT, NODE_EMBED, NODE_SEARCH, NODE_ANSWER, NODE_FINALIZE]
