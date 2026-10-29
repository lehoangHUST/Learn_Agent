"""Test phân loại intent (rule) và các conditional edge.

Cả hai đều là hàm thuần, nên test chạy trong micro-giây và không cần dựng gì. Đó
chính là lý do intent được làm bằng rule chứ không bằng LLM: phần lớn quyết định ở
đây là tất định, và một hàm tất định thì test được đến từng nhánh.
"""

from __future__ import annotations

import pytest

from app.agent.nodes.intent import classify_by_rules, extract_face_indices, normalize
from app.agent.routing import (
    NODE_ANALYZE,
    NODE_ANSWER,
    NODE_DETECT,
    NODE_EMBED,
    NODE_ERROR,
    NODE_FINALIZE,
    NODE_INTENT,
    NODE_PLAN,
    NODE_SEARCH,
    next_planned_step,
    route_after_analyze,
    route_after_answer,
    route_after_detect,
    route_after_error,
    route_after_intent,
    route_after_plan,
    route_after_validate,
)
from app.agent.state import (
    INTENT_DETECTION_AND_SEARCH,
    INTENT_FACE_DETECTION,
    INTENT_FACE_SEARCH,
    INTENT_GENERAL_FACE_QUESTION,
    INTENT_RESULT_QUESTION,
    INTENT_UNSUPPORTED,
)


# --------------------------------------------------------------------------- #
# Chuẩn hoá & trích chỉ số
# --------------------------------------------------------------------------- #
def test_normalize_strips_accents() -> None:
    assert normalize("Có bao nhiêu KHUÔN MẶT?") == "co bao nhieu khuon mat?"


def test_normalize_handles_d_with_stroke() -> None:
    """``đ`` không có dạng phân rã NFD nên phải map tay.

    Thiếu bước này thì mọi từ khoá có "đ" - "đếm", "đối chiếu", "độ tuổi" - sẽ không
    bao giờ khớp, và cái sai đó im lặng: câu hỏi vẫn được phân loại, chỉ là sai nhãn.
    """
    assert normalize("Đếm khuôn mặt") == "dem khuon mat"
    assert normalize("đối chiếu") == "doi chieu"


@pytest.mark.parametrize(
    "query,expected_intent",
    [
        ("Đếm khuôn mặt trong ảnh", INTENT_FACE_DETECTION),
        ("Đối chiếu khuôn mặt này với dữ liệu", INTENT_FACE_SEARCH),
        ("Người này bao nhiêu độ tuổi", INTENT_UNSUPPORTED),
    ],
)
def test_keywords_with_d_stroke_classify_correctly(query: str, expected_intent: str) -> None:
    decision = classify_by_rules(
        query, has_new_images=True, prior_detection=False, prior_search=False
    )
    assert decision.intent == expected_intent


@pytest.mark.parametrize(
    "query,expected",
    [
        ("Tìm kiếm khuôn mặt thứ hai", [1]),
        ("tim khuon mat thu 3", [2]),
        ("khuôn mặt thứ nhất là ai", [0]),
        ("mặt đầu tiên", [0]),
        ("face thu 2", [1]),
        ("khuôn mặt 2", [1]),
        ("mặt số 3", [2]),
        ("face_index=0", [0]),
        ("face_index: 4", [4]),
        ("khuôn mặt to nhất", [0]),
    ],
)
def test_extract_face_indices(query: str, expected: list[int]) -> None:
    indices, _ = extract_face_indices(normalize(query))
    assert indices == expected


def test_result_ordinal_is_not_mistaken_for_face_ordinal() -> None:
    """"kết quả thứ hai" nói về ứng viên rank 2, KHÔNG phải khuôn mặt index 1.

    Gộp hai thứ này lại là tìm sai khuôn mặt - một lỗi rất dễ xảy ra nếu chỉ regex
    'thứ (\\d+)' mà không gắn với từ chỉ khuôn mặt.
    """
    indices, _ = extract_face_indices(normalize("Kết quả thứ hai có similarity bao nhiêu?"))
    assert indices is None


def test_no_face_reference_returns_none() -> None:
    indices, _ = extract_face_indices(normalize("Có bao nhiêu khuôn mặt trong ảnh?"))
    assert indices is None


# --------------------------------------------------------------------------- #
# Phân loại intent
# --------------------------------------------------------------------------- #
def test_unsupported_wins_over_everything() -> None:
    """Yêu cầu suy đoán thuộc tính bị từ chối trước cả khi xét tới ảnh."""
    for query in (
        "Người trong ảnh bao nhiêu tuổi?",
        "Giới tính của khuôn mặt này là gì?",
        "Nhận dạng cảm xúc của người này",
        "theo dõi người này trong camera",
    ):
        decision = classify_by_rules(
            query, has_new_images=True, prior_detection=False, prior_search=False
        )
        assert decision.intent == INTENT_UNSUPPORTED, query
        assert decision.confident is True


def test_detection_with_image() -> None:
    decision = classify_by_rules(
        "Có bao nhiêu khuôn mặt trong ảnh?",
        has_new_images=True,
        prior_detection=False,
        prior_search=False,
    )
    assert decision.intent == INTENT_FACE_DETECTION
    assert decision.confident is True


def test_search_with_image() -> None:
    decision = classify_by_rules(
        "Tìm kiếm khuôn mặt trong ảnh này",
        has_new_images=True,
        prior_detection=False,
        prior_search=False,
    )
    assert decision.intent == INTENT_FACE_SEARCH


def test_detection_and_search_when_both_asked() -> None:
    decision = classify_by_rules(
        "Phát hiện khuôn mặt rồi tìm kiếm xem là ai",
        has_new_images=True,
        prior_detection=False,
        prior_search=False,
    )
    assert decision.intent == INTENT_DETECTION_AND_SEARCH


def test_ambiguous_query_with_image_defaults_to_cheapest_useful_action() -> None:
    """Có ảnh nhưng câu không rõ -> chọn detection (rẻ hơn) và tự nhận là không chắc."""
    decision = classify_by_rules(
        "Xem ảnh này", has_new_images=True, prior_detection=False, prior_search=False
    )
    assert decision.intent == INTENT_FACE_DETECTION
    assert decision.confident is False  # -> mở đường cho LLM xem lại nếu được bật


def test_followup_search_without_new_image_uses_prior_detection() -> None:
    decision = classify_by_rules(
        "Tìm kiếm khuôn mặt thứ hai",
        has_new_images=False,
        prior_detection=True,
        prior_search=False,
    )
    assert decision.intent == INTENT_FACE_SEARCH
    assert decision.confident is True
    assert decision.face_indices == [1]


def test_result_question_needs_prior_results() -> None:
    decision = classify_by_rules(
        "Kết quả thứ hai có similarity bao nhiêu?",
        has_new_images=False,
        prior_detection=True,
        prior_search=True,
    )
    assert decision.intent == INTENT_RESULT_QUESTION


def test_general_question_without_any_data() -> None:
    decision = classify_by_rules(
        "Giải thích sự khác biệt giữa confidence của detector và similarity score",
        has_new_images=False,
        prior_detection=False,
        prior_search=False,
    )
    assert decision.intent == INTENT_GENERAL_FACE_QUESTION


def test_why_not_found_is_a_result_question() -> None:
    decision = classify_by_rules(
        "Tại sao hệ thống không tìm thấy kết quả phù hợp?",
        has_new_images=False,
        prior_detection=True,
        prior_search=True,
    )
    assert decision.intent == INTENT_RESULT_QUESTION


def test_signals_are_reported_for_debuggability() -> None:
    decision = classify_by_rules(
        "tìm kiếm", has_new_images=True, prior_detection=False, prior_search=False
    )
    assert decision.signals["search_hits"]
    assert decision.signals["has_new_images"] is True
    assert decision.reason


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def plan(*steps: str) -> dict:
    return {"execution_plan": [{"step": s, "reason": ""} for s in steps]}


def failed_at(node: str) -> dict:
    """State sau khi ``node`` vừa chạy và lỗi.

    ``failed_node`` là tín hiệu router đọc; ``errors`` là lịch sử. Hai thứ khác nhau,
    và test dựng cả hai đúng như decorator ``graph_node`` sinh ra.
    """
    return {
        "failed_node": node,
        "errors": [{"code": "x", "message": "m", "retryable": False, "node": node}],
    }


def succeeded_after_failure_at(node: str) -> dict:
    """State sau khi ``node`` lỗi rồi retry THÀNH CÔNG.

    ``errors`` vẫn giữ lỗi cũ (lịch sử), nhưng ``failed_node`` đã được xoá.
    """
    return {
        "failed_node": None,
        "errors": [{"code": "x", "message": "m", "retryable": True, "node": node}],
    }


def test_next_planned_step_walks_the_plan() -> None:
    state = plan(NODE_DETECT, NODE_EMBED, NODE_SEARCH, NODE_ANALYZE, NODE_ANSWER)
    assert next_planned_step(state, None) == NODE_DETECT
    assert next_planned_step(state, NODE_DETECT) == NODE_EMBED
    assert next_planned_step(state, NODE_SEARCH) == NODE_ANALYZE
    assert next_planned_step(state, NODE_ANSWER) == NODE_FINALIZE


def test_next_planned_step_skips_absent_steps() -> None:
    """Kế hoạch detection-only: sau detect là analyze, bỏ qua embed/search."""
    state = plan(NODE_DETECT, NODE_ANALYZE, NODE_ANSWER)
    assert next_planned_step(state, NODE_DETECT) == NODE_ANALYZE


def test_empty_plan_goes_to_finalize() -> None:
    assert next_planned_step({}, None) == NODE_FINALIZE


def test_node_outside_plan_goes_to_finalize() -> None:
    """answer được gọi từ handle_error, ngoài kế hoạch -> không đi lạc vào plan."""
    state = plan(NODE_DETECT, NODE_ANALYZE)
    assert next_planned_step(state, NODE_ANSWER) == NODE_FINALIZE


def test_routers_send_failures_to_error_handler() -> None:
    assert route_after_validate(failed_at("validate_input")) == NODE_ERROR
    assert route_after_intent(failed_at("classify_intent")) == NODE_ERROR
    assert route_after_plan(failed_at("plan_execution")) == NODE_ERROR
    assert route_after_detect({**plan(NODE_DETECT), **failed_at(NODE_DETECT)}) == NODE_ERROR
    assert route_after_analyze({**plan(NODE_ANALYZE), **failed_at(NODE_ANALYZE)}) == NODE_ERROR


def test_router_ignores_errors_from_earlier_nodes() -> None:
    """Lỗi cũ (đã xử lý) không được đẩy node sau về handle_error -> chống vòng lặp."""
    state = {**plan(NODE_DETECT, NODE_ANALYZE, NODE_ANSWER), **failed_at("validate_input")}
    assert route_after_detect(state) == NODE_ANALYZE


def test_router_continues_after_a_successful_retry() -> None:
    """Sau khi retry thành công, lỗi cũ vẫn trong ``errors`` nhưng luồng phải đi tiếp.

    Nếu router suy "ai vừa lỗi" từ ``errors[-1].node`` thì ở đây nó lại thấy
    ``detect_faces`` và đẩy về handle_error - vòng lặp retry không bao giờ dứt.
    """
    state = {
        **plan(NODE_DETECT, NODE_ANALYZE, NODE_ANSWER),
        **succeeded_after_failure_at(NODE_DETECT),
    }
    assert route_after_detect(state) == NODE_ANALYZE


def test_route_after_answer_always_finalizes() -> None:
    assert route_after_answer({}) == NODE_FINALIZE
    assert route_after_answer(failed_at(NODE_ANSWER)) == NODE_FINALIZE


def test_route_after_error_retries_when_target_set() -> None:
    assert route_after_error({"retry_target": NODE_DETECT}) == NODE_DETECT
    assert route_after_error({"retry_target": NODE_SEARCH}) == NODE_SEARCH


def test_route_after_error_explains_when_no_retry() -> None:
    state = {"retry_target": None, **failed_at(NODE_DETECT)}
    assert route_after_error(state) == NODE_ANSWER


def test_route_after_error_finalizes_if_answer_already_exists() -> None:
    state = {"retry_target": None, "answer": "đã trả lời", **failed_at(NODE_DETECT)}
    assert route_after_error(state) == NODE_FINALIZE


def test_route_after_error_does_not_loop_back_into_answer() -> None:
    """Chính node answer lỗi -> đi finalize, không gửi lại vào answer."""
    state = {"retry_target": None, **failed_at(NODE_ANSWER)}
    assert route_after_error(state) == NODE_FINALIZE


def test_route_after_error_rejects_unknown_retry_target() -> None:
    """retry_target không phải node hợp lệ -> không nhảy bừa."""
    state = {"retry_target": "nonexistent_node", **failed_at(NODE_DETECT)}
    assert route_after_error(state) in (NODE_ANSWER, NODE_FINALIZE)


def test_happy_path_routing_is_linear() -> None:
    state = plan(NODE_DETECT, NODE_EMBED, NODE_SEARCH, NODE_ANALYZE, NODE_ANSWER)
    assert route_after_validate(state) == NODE_INTENT
    assert route_after_intent(state) == NODE_PLAN
    assert route_after_plan(state) == NODE_DETECT
    assert route_after_detect(state) == NODE_EMBED
    assert route_after_analyze(state) == NODE_ANSWER
