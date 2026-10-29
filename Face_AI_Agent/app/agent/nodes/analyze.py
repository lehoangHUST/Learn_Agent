"""Node ``analyze_results``: tổng hợp dữ kiện và dựng ``grounding``.

``grounding`` là **toàn bộ** những gì node trả lời được phép dùng. Mọi con số trong
câu trả lời cuối cùng đều phải truy ngược về được một field ở đây. Tách bước này
thành một node riêng đem lại ba thứ:

1. Một chỗ duy nhất để kiểm tra "dữ liệu đã đủ để trả lời chưa".
2. Một ranh giới rõ giữa *đo được* và *diễn đạt*: trường ``provenance`` ghi thẳng
   nguồn gốc của từng phần, nên khi đọc trace Langfuse ta biết ngay con số nào do
   detector/FAISS tạo ra và phần nào do LLM viết.
3. Chỗ để tính các dữ kiện phái sinh một cách tất định (mặt lớn nhất, đếm theo
   match_status, cách biệt top1-top2) - thay vì để LLM tự "nhìn" ra từ JSON, việc
   mà nó làm sai một cách im lặng.

Node này cũng xử lý nhánh ``result_question``: khi lượt này không chạy model, dữ
liệu được **đọc lại từ result store** theo ``target_result_id``, luôn qua principal.
"""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from app.agent.nodes.common import graph_node
from app.agent.prompts import GENERAL_KNOWLEDGE_NOTES, MATCH_STATUS_EXPLANATION
from app.agent.state import (
    INTENT_GENERAL_FACE_QUESTION,
    FaceAgentContext,
    FaceAgentState,
)
from app.errors import ResultNotFound
from app.repositories.result_store import KIND_DETECTION, KIND_SEARCH


@graph_node("analyze_results")
def analyze_results(state: FaceAgentState, runtime: Runtime[FaceAgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    store = ctx.deps.results

    detection = state.get("detection_summary")
    search = state.get("search_summary")
    warnings: list[str] = []
    loaded_from: list[str] = []

    # Nhánh hỏi về kết quả cũ: lấy lại từ kho, không chạy lại model.
    target_id = state.get("target_result_id")
    if target_id and not (detection or search):
        try:
            stored = store.get(target_id, principal=ctx.principal)
        except ResultNotFound as exc:
            # Không raise: thiếu dữ liệu là một câu trả lời hợp lệ ("kết quả đã hết
            # hạn"), và node answer sẽ giải thích. Raise ở đây sẽ biến nó thành lỗi
            # hệ thống trong khi nó là tình trạng bình thường của dữ liệu có TTL.
            warnings.append(f"không đọc được kết quả {target_id}: {exc.message}")
            stored = None
        if stored is not None:
            loaded_from.append(stored.result_id)
            if stored.kind == KIND_SEARCH:
                search = stored.payload
                source_det = stored.payload.get("source_detection_result_id")
                if source_det:
                    try:
                        detection = store.get(
                            source_det, principal=ctx.principal, kind=KIND_DETECTION
                        ).payload
                        loaded_from.append(source_det)
                    except ResultNotFound:
                        warnings.append(
                            "kết quả detection gốc của lần search này đã hết hạn; "
                            "chỉ còn dữ liệu search"
                        )
            else:
                detection = stored.payload

    analysis: dict[str, Any] = {
        "has_detection_data": detection is not None,
        "has_search_data": search is not None,
        "loaded_from_result_ids": loaded_from,
    }

    if detection:
        analysis["detection"] = _analyze_detection(detection)
    if search:
        analysis["search"] = _analyze_search(search)
        warnings.extend(
            w for w in search.get("warnings", []) if w not in (state.get("warnings") or [])
        )

    analysis["sufficient_for_answer"] = bool(
        detection or search or state.get("intent") == INTENT_GENERAL_FACE_QUESTION
    )
    if not analysis["sufficient_for_answer"]:
        analysis["missing_data"] = [
            "chưa có kết quả detection hoặc search nào để trả lời câu hỏi này"
        ]

    grounding: dict[str, Any] = {
        "query": state.get("query"),
        "intent": state.get("intent"),
        "intent_source": state.get("intent_source"),
        "capabilities": state.get("capabilities") or {},
        "detection_result": detection,
        "search_result": search,
        "analysis": analysis,
        "errors": list(state.get("errors") or []),
        "warnings": list(state.get("warnings") or []) + warnings,
        "concept_notes": GENERAL_KNOWLEDGE_NOTES,
        "match_status_meaning": MATCH_STATUS_EXPLANATION,
        # Ghi rõ cái gì từ đâu. Đây là field mà người đọc trace dùng để biết câu trả
        # lời có bị LLM "thêm" gì vào không.
        "provenance": {
            "detection_result": "đo bởi face detector (box, landmark); detector không trả confidence",
            "search_result": "tính bởi vector search trên FAISS (cosine similarity)",
            "analysis": "tính tất định bằng code từ hai nguồn trên",
            "answer_text": "do LLM hoặc template diễn đạt lại, KHÔNG được thêm số mới",
        },
    }

    return {
        "analysis": analysis,
        "grounding": grounding,
        "detection_summary": detection,
        "search_summary": search,
        "warnings": warnings,
    }


def _analyze_detection(payload: dict[str, Any]) -> dict[str, Any]:
    faces = payload.get("faces") or []
    out: dict[str, Any] = {
        "image_count": payload.get("image_count"),
        "total_face_count": payload.get("total_face_count", len(faces)),
        "usable_face_count": payload.get("usable_face_count"),
        "detector_confidence_available": payload.get("detector_confidence_available", False),
    }

    if not faces:
        out["note"] = (
            "không phát hiện được khuôn mặt nào - không có dữ liệu để so khớp hay "
            "trả lời về khuôn mặt"
        )
        return out

    # Các mặt đã được sắp theo diện tích giảm dần ở tầng vision, nhưng tính lại ở
    # đây để dữ kiện này không phụ thuộc vào thứ tự của payload.
    by_area = sorted(
        faces,
        key=lambda f: (f.get("size") or {}).get("area_px") or 0.0,
        reverse=True,
    )
    largest = by_area[0]
    out["largest_face"] = {
        "global_face_index": largest.get("global_face_index"),
        "area_px": (largest.get("size") or {}).get("area_px"),
        "box": largest.get("box"),
    }
    out["face_size_ranking"] = [
        {
            "global_face_index": f.get("global_face_index"),
            "area_px": (f.get("size") or {}).get("area_px"),
        }
        for f in by_area
    ]
    out["size_is_not_confidence"] = (
        "thứ tự theo diện tích chỉ nói khuôn mặt nào to hơn trong ảnh; nó KHÔNG phải "
        "confidence của detector và không nói gì về việc phát hiện có đúng hay không"
    )
    rejected = [f for f in faces if not f.get("usable_for_search")]
    if rejected:
        out["rejected_faces"] = [
            {
                "global_face_index": f.get("global_face_index"),
                "reason": f.get("rejected_reason"),
            }
            for f in rejected
        ]
    return out


def _analyze_search(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results") or []
    threshold = payload.get("threshold") or {}

    status_counts: dict[str, int] = {}
    for item in results:
        status = item.get("match_status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    out: dict[str, Any] = {
        "query_face_count": payload.get("query_face_count", len(results)),
        "status_counts": status_counts,
        "threshold": threshold,
        "threshold_calibrated": threshold.get("calibrated", False),
        "index_total_faces": (payload.get("index") or {}).get("total_faces"),
        "index_total_subjects": (payload.get("index") or {}).get("total_subjects"),
        "per_face": [
            {
                "face_index": item.get("face_index"),
                "match_status": item.get("match_status"),
                "best_score": item.get("best_score"),
                "runner_up_score": item.get("runner_up_score"),
                "margin": item.get("margin"),
                "candidate_count": len(item.get("candidates") or []),
                "best_subject_id": (
                    (item.get("candidates") or [{}])[0].get("subject_id")
                    if item.get("candidates")
                    else None
                ),
                "reason": item.get("reason"),
            }
            for item in results
        ],
    }

    if not threshold.get("calibrated", False):
        out["calibration_warning"] = (
            "ngưỡng so khớp CHƯA được hiệu chuẩn cho model này: mọi trạng thái 'match' "
            "chỉ nên coi là ứng viên cần người xác nhận, không phải kết luận danh tính"
        )

    if (payload.get("index") or {}).get("total_faces") == 0:
        out["empty_index"] = (
            "collection chưa có khuôn mặt nào được đăng ký, nên không thể có kết quả "
            "so khớp - đây KHÁC với 'đã tìm và không thấy ai giống'"
        )
    return out
