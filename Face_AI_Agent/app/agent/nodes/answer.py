"""Node ``answer_question``.

Hai đường trả lời, **cùng một nguồn dữ liệu**:

* ``render_template_answer`` - tất định, không gọi mạng, không tốn tiền. Đây là
  đường mặc định và cũng là đường dự phòng.
* LLM - diễn đạt lại cùng ``grounding`` cho mượt hơn.

Điểm quan trọng: con số trong hai chế độ phải *giống nhau*, vì cả hai đọc chung
``grounding``. LLM không được thêm, bớt hay làm tròn số. Nếu LLM lỗi hay bị tắt,
người dùng vẫn nhận được câu trả lời đầy đủ dữ kiện - chỉ là văn phong khô hơn.
Đó là lý do template được viết đầy đủ chứ không phải một dòng "LLM unavailable".

``render_template_answer`` là hàm thuần -> test được bằng cách dựng ``grounding``
bằng tay, không cần LLM, không cần graph, không cần backend.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from app.agent.nodes.common import graph_node
from app.agent.prompts import GENERAL_KNOWLEDGE_NOTES, MATCH_STATUS_EXPLANATION
from app.agent.state import (
    INTENT_GENERAL_FACE_QUESTION,
    AnswerNodeInput,
    FaceAgentContext,
)
from app.errors import LLMError

logger = logging.getLogger(__name__)


def _fmt_score(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def render_template_answer(query: str, grounding: dict[str, Any]) -> str:
    """Sinh câu trả lời tất định từ ``grounding``. Không bịa gì."""
    lines: list[str] = []
    analysis = grounding.get("analysis") or {}
    detection = grounding.get("detection_result")
    search = grounding.get("search_result")
    errors = grounding.get("errors") or []
    intent = grounding.get("intent")

    # 1. Lỗi/thiếu dữ liệu nói trước, vì đó là câu trả lời thật sự cho người dùng.
    blocking = [e for e in errors if not e.get("retryable")]
    if blocking:
        lines.append("Không thực hiện được yêu cầu:")
        for err in blocking:
            lines.append(f"- [{err.get('code')}] {err.get('message')}")
        missing = analysis.get("missing_data")
        if missing:
            lines.append("Dữ liệu còn thiếu: " + "; ".join(missing))
        if not (detection or search):
            return "\n".join(lines)
        lines.append("")

    if intent == INTENT_GENERAL_FACE_QUESTION and not (detection or search):
        lines.append("Giải thích các khái niệm trong hệ thống này:")
        lines.append(GENERAL_KNOWLEDGE_NOTES.strip())
        return "\n".join(lines)

    # 2. Detection
    if detection:
        det = analysis.get("detection") or {}
        total = det.get("total_face_count", detection.get("total_face_count", 0))
        images = det.get("image_count", detection.get("image_count", 1))
        usable = det.get("usable_face_count", detection.get("usable_face_count"))

        if total == 0:
            lines.append(f"Không phát hiện được khuôn mặt nào trong {images} ảnh.")
            for warning in detection.get("warnings", [])[:3]:
                lines.append(f"- {warning}")
        else:
            lines.append(
                f"Phát hiện {total} khuôn mặt trong {images} ảnh"
                + (f" ({usable} khuôn mặt dùng được cho tìm kiếm)." if usable is not None else ".")
            )
            for face in detection.get("faces", []):
                box = face.get("box")
                size = face.get("size") or {}
                idx = face.get("global_face_index", face.get("face_index"))
                note = "" if face.get("usable_for_search") else f" [loại: {face.get('rejected_reason')}]"
                lines.append(
                    f"- Khuôn mặt #{idx}: box={box}, "
                    f"kích thước {size.get('width_px')}x{size.get('height_px')}px"
                    f"{note}"
                )
            if not detection.get("detector_confidence_available", False):
                lines.append(
                    "- Lưu ý: detector của backend này KHÔNG trả confidence score, nên "
                    "không thể xếp hạng khuôn mặt theo độ tin cậy. Chỉ có kích thước "
                    "bounding box, và kích thước không phải confidence."
                )

    # 3. Search
    if search:
        if lines:
            lines.append("")
        srch = analysis.get("search") or {}
        threshold = (search.get("threshold") or {}).get("match_threshold")
        index_info = search.get("index") or {}
        lines.append(
            f"Kết quả tìm kiếm trong collection {search.get('collection')!r} "
            f"({index_info.get('total_faces', 0)} khuôn mặt của "
            f"{index_info.get('total_subjects', 0)} người đã đăng ký), "
            f"ngưỡng = {_fmt_score(threshold)}:"
        )

        results = search.get("results") or []
        if not results:
            lines.append("- Không có khuôn mặt nào dùng được để truy vấn.")
        for item in results:
            status = item.get("match_status")
            lines.append(
                f"- Khuôn mặt #{item.get('face_index')}: {status} "
                f"(top1={_fmt_score(item.get('best_score'))}, "
                f"top2={_fmt_score(item.get('runner_up_score'))}, "
                f"cách biệt={_fmt_score(item.get('margin'))})"
            )
            meaning = MATCH_STATUS_EXPLANATION.get(str(status))
            if meaning:
                lines.append(f"  Nghĩa là: {meaning}.")
            if item.get("reason"):
                lines.append(f"  Căn cứ: {item['reason']}")
            for cand in (item.get("candidates") or [])[:5]:
                meta = cand.get("metadata") or {}
                meta_text = ""
                if not cand.get("metadata_visible", True):
                    meta_text = " (metadata bị ẩn do thiếu quyền)"
                elif meta:
                    meta_text = f" metadata={meta}"
                lines.append(
                    f"    #{cand.get('rank')} subject={cand.get('subject_id')} "
                    f"similarity={_fmt_score(cand.get('similarity_score'))}"
                    f"{meta_text}"
                )

        if srch.get("empty_index"):
            lines.append(f"- {srch['empty_index']}")
        if srch.get("calibration_warning"):
            lines.append(f"- CẢNH BÁO: {srch['calibration_warning']}")
        lines.append(
            "- similarity_score là cosine similarity trong [-1, 1], không phải xác "
            "suất cùng danh tính."
        )

    if not lines:
        lines.append(
            "Chưa có dữ liệu detection hoặc search nào để trả lời. Hãy gửi kèm một ảnh."
        )

    warnings = grounding.get("warnings") or []
    if warnings:
        lines.append("")
        lines.append("Ghi chú:")
        for warning in dict.fromkeys(warnings):  # bỏ trùng, giữ thứ tự
            lines.append(f"- {warning}")

    return "\n".join(lines)


@graph_node("answer_question")
def answer_question(state: AnswerNodeInput, runtime: Runtime[FaceAgentContext]) -> dict[str, Any]:
    """Node này khai báo ``input_schema=AnswerNodeInput`` khi đăng ký vào graph.

    Nó chỉ nhìn thấy query, intent, grounding, analysis, errors, warnings - không
    thấy image_refs, user_id hay bất kỳ field nội bộ nào khác. Thu hẹp tầm nhìn của
    node duy nhất gọi LLM là thu hẹp đúng chỗ dữ liệu có thể rò ra prompt.
    """
    ctx = runtime.context
    query = state.get("query") or ""
    grounding = dict(state.get("grounding") or {})

    # Node answer có thể được gọi từ handle_error, khi analyze chưa kịp chạy.
    grounding.setdefault("intent", state.get("intent"))
    grounding.setdefault("analysis", state.get("analysis") or {})
    grounding.setdefault("errors", list(state.get("errors") or []))
    grounding.setdefault("warnings", list(state.get("warnings") or []))
    grounding.setdefault("match_status_meaning", MATCH_STATUS_EXPLANATION)

    template_answer = render_template_answer(query, grounding)
    answer = template_answer
    source = "template"
    warnings: list[str] = []

    llm = ctx.deps.llm
    if llm is not None and ctx.deps.settings.llm.use_for_answer:
        try:
            answer = llm.answer(query, grounding)
            source = "llm"
        except LLMError as exc:
            # Không để lỗi LLM thành lỗi request: template đã có đủ dữ kiện.
            warnings.append(
                f"LLM không trả lời được ({exc.code}); dùng câu trả lời dạng bảng dữ liệu"
            )
            logger.warning("answer_llm_failed", extra={"error_code": exc.code})

    return {
        "answer": answer,
        "answer_source": source,
        "messages": [AIMessage(content=answer)],
        "grounding": {**grounding, "answer_source": source},
        "warnings": warnings,
    }
