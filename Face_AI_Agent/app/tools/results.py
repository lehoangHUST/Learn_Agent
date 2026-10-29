"""Tool: đọc lại kết quả đã lưu và giải thích kết quả so khớp.

``explain_matching_result`` là hàm **thuần**: nhận payload của một kết quả search và
trả về lời giải thích có cấu trúc. Nó được dùng ở hai nơi - endpoint
``/results/{id}/explain`` và phần giải thích trong câu trả lời - nên lời giải thích
là một, không phải hai bản viết tay dễ lệch nhau.

Mọi hàm đọc ở đây đều nhận ``Principal``: không có đường nào đọc kết quả mà không
qua kiểm quyền.
"""

from __future__ import annotations

from typing import Any

from app.agent.prompts import MATCH_STATUS_EXPLANATION
from app.dependencies import AgentDeps
from app.errors import ResultNotFound
from app.repositories.result_store import KIND_DETECTION, KIND_SEARCH, StoredResult
from app.security import Principal


def get_detection_result(
    result_id: str, *, deps: AgentDeps, principal: Principal
) -> dict[str, Any]:
    stored = deps.results.get(result_id, principal=principal, kind=KIND_DETECTION)
    return _envelope(stored)


def get_face_search_result(
    result_id: str, *, deps: AgentDeps, principal: Principal
) -> dict[str, Any]:
    stored = deps.results.get(result_id, principal=principal, kind=KIND_SEARCH)
    return _envelope(stored)


def get_result(result_id: str, *, deps: AgentDeps, principal: Principal) -> dict[str, Any]:
    """Đọc kết quả bất kể loại."""
    return _envelope(deps.results.get(result_id, principal=principal))


def list_thread_results(
    thread_id: str, *, deps: AgentDeps, principal: Principal, limit: int = 20
) -> list[dict[str, Any]]:
    return [
        {
            "result_id": r.result_id,
            "kind": r.kind,
            "created_at": r.created_at,
            "expires_at": r.expires_at,
            "age_seconds": round(r.age_seconds, 1),
        }
        for r in deps.results.list_for_thread(
            thread_id=thread_id, principal=principal, limit=limit
        )
    ]


def explain_matching_result(
    payload: dict[str, Any],
    *,
    face_index: int | None = None,
) -> dict[str, Any]:
    """Giải thích vì sao một kết quả search ra trạng thái như vậy.

    Chỉ dùng số đã có trong ``payload``. Không gọi model, không gọi LLM, không tính
    lại similarity - nên lời giải thích không thể lệch khỏi con số đã trả cho người
    dùng trước đó.
    """
    threshold_info = payload.get("threshold") or {}
    threshold = threshold_info.get("match_threshold")
    review_band = threshold_info.get("review_band") or 0.0
    margin_cfg = threshold_info.get("ambiguous_margin") or 0.0

    results = payload.get("results") or []
    if face_index is not None:
        results = [r for r in results if r.get("face_index") == face_index]
        if not results:
            raise ResultNotFound(
                f"kết quả này không có khuôn mặt index {face_index}",
                details={
                    "available_face_indices": [
                        r.get("face_index") for r in (payload.get("results") or [])
                    ]
                },
            )

    explanations: list[dict[str, Any]] = []
    for item in results:
        status = str(item.get("match_status"))
        best = item.get("best_score")
        runner_up = item.get("runner_up_score")
        candidates = item.get("candidates") or []

        explanations.append(
            {
                "face_index": item.get("face_index"),
                "match_status": status,
                "meaning": MATCH_STATUS_EXPLANATION.get(status, "trạng thái không rõ"),
                "numbers": {
                    "best_score": best,
                    "runner_up_score": runner_up,
                    "margin": item.get("margin"),
                    "match_threshold": threshold,
                    "review_band_lower_bound": (
                        None if threshold is None else round(threshold - review_band, 6)
                    ),
                    "ambiguous_margin": margin_cfg,
                },
                "decision_rule": item.get("reason"),
                "candidate_count": len(candidates),
                "top_candidates": [
                    {
                        "rank": c.get("rank"),
                        "subject_id": c.get("subject_id"),
                        "similarity_score": c.get("similarity_score"),
                    }
                    for c in candidates[:5]
                ],
                "what_would_change_it": _what_would_change(status, best, threshold, review_band),
            }
        )

    return {
        "threshold": threshold_info,
        "threshold_calibrated": threshold_info.get("calibrated", False),
        "index": payload.get("index"),
        "explanations": explanations,
        "score_semantics": (
            "similarity_score là cosine similarity trong [-1, 1] giữa hai embedding. "
            "Nó KHÔNG phải xác suất cùng danh tính và không nên chuyển thành phần trăm."
        ),
        "caveat": (
            "ngưỡng chưa hiệu chuẩn thì kết quả 'match' chỉ là ứng viên cần người xác nhận"
            if not threshold_info.get("calibrated", False)
            else "ngưỡng đã được hiệu chuẩn trên tập validation"
        ),
    }


def _what_would_change(
    status: str, best: float | None, threshold: float | None, review_band: float
) -> str:
    """Nói cho người dùng biết cần gì để kết quả khác đi - thay vì chỉ 'không tìm thấy'."""
    if best is None or threshold is None:
        return (
            "chưa có ứng viên nào để so sánh: cần đăng ký khuôn mặt vào collection, "
            "hoặc nới bộ lọc metadata/quyền truy cập"
        )
    gap = threshold - best
    if status == "match":
        return f"nếu ngưỡng tăng thêm {max(gap, 0):.4f} nữa thì kết quả này sẽ không còn là match"
    if status in ("needs_review", "no_match"):
        return (
            f"cần thêm {gap:.4f} similarity nữa để vượt ngưỡng {threshold:.4f}. "
            "Nguyên nhân thường gặp: ảnh truy vấn khác góc/ánh sáng/độ nét so với ảnh "
            "đã đăng ký, hoặc người này chưa có trong collection"
        )
    if status == "ambiguous":
        return (
            "cần tăng cách biệt giữa ứng viên thứ nhất và thứ hai: thường là do trong "
            "collection có nhiều ảnh rất giống nhau, hoặc có bản ghi trùng của cùng một người"
        )
    return "không xác định"


def _envelope(stored: StoredResult) -> dict[str, Any]:
    return {
        "result_id": stored.result_id,
        "kind": stored.kind,
        "thread_id": stored.thread_id,
        "request_id": stored.request_id,
        "created_at": stored.created_at,
        "expires_at": stored.expires_at,
        "age_seconds": round(stored.age_seconds, 1),
        "payload": stored.payload,
    }
