"""Endpoint ``/api/v1/agent/chat`` - đường đi qua LangGraph.

Hỗ trợ hỏi tiếp: truyền lại ``thread_id`` của lượt trước. Nhưng cần phân biệt rõ
hai thứ mà ``thread_id`` kéo theo:

* **Lịch sử hội thoại** (messages, nằm trong checkpoint của LangGraph) - dùng cho
  ngữ cảnh đối thoại.
* **Kết quả máy học** (nằm trong result store, có owner + TTL) - dùng cho mọi con số.

Agent **không** đọc số từ lịch sử hội thoại. Nếu nó làm vậy thì ở lượt thứ ba nó sẽ
đang đọc lại văn bản mà chính nó viết ở lượt thứ hai - sai số tự nhân lên và không
còn nguồn nào để đối chiếu.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import DepsDep, RateLimitedPrincipalDep, RunnerDep
from app.logging_utils import new_request_id
from app.schemas.agent import ChatRequest, ChatResponse
from app.schemas.common import ErrorResponse
from app.security import ROLE_SEARCH

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
    summary="Hỏi bằng ngôn ngữ tự nhiên, kèm ảnh tuỳ chọn",
)
def chat_endpoint(
    body: ChatRequest,
    deps: DepsDep,
    principal: RateLimitedPrincipalDep,
    runner: RunnerDep,
) -> ChatResponse:
    """Chạy workflow LangGraph cho một lượt hội thoại.

    Response luôn có ``thread_id``: truyền lại ở lượt sau để hỏi tiếp về cùng kết quả.

    ``status`` = ``partial`` nghĩa là có kết quả dùng được **và** có lỗi - đọc cả
    ``errors`` trước khi dùng dữ liệu.
    """
    principal.require_role(ROLE_SEARCH)
    request_id = new_request_id()

    result = runner.run(
        principal=principal,
        query=body.message,
        images_base64=body.images_base64 or None,
        collection=body.collection,
        thread_id=body.thread_id,
        request_id=request_id,
        face_indices=body.face_indices,
        target_result_id=body.target_result_id,
    )

    output = result.output
    return ChatResponse(
        request_id=result.request_id,
        thread_id=result.thread_id,
        status=output.get("status", "failed"),
        intent=output.get("intent", "unsupported_request"),
        intent_source=output.get("intent_source", "rule"),
        answer=output.get("answer", ""),
        # Nói rõ câu trả lời do template hay LLM diễn đạt. Người đọc có quyền biết
        # phần văn bản nào do model sinh ra.
        answer_source=output.get("answer_source") or "template",
        detection_result_id=output.get("detection_result_id"),
        search_result_id=output.get("search_result_id"),
        detection_summary=output.get("detection_summary"),
        search_summary=output.get("search_summary"),
        analysis=output.get("analysis"),
        executed_steps=output.get("executed_steps") or [],
        errors=output.get("errors") or [],
        warnings=output.get("warnings") or [],
    )
