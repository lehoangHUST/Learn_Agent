"""Endpoint tra cứu kết quả: ``/tasks/{request_id}``, ``/results/*``, ``/threads/*``.

Mọi endpoint ở đây đi qua ``Principal``. Đây là chỗ mà bất biến "không đọc được kết
quả của session khác" được kiểm thật: ``result_id`` là chuỗi đoán được về mặt lý
thuyết, nên quyền sở hữu phải được kiểm ở tầng dữ liệu, không phải dựa vào việc id
khó đoán.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.deps import DepsDep, PrincipalDep
from app.schemas.agent import (
    ExplainResponse,
    TaskStatusResponse,
    ThreadResultsResponse,
)
from app.schemas.common import ErrorResponse
from app.tools.results import (
    explain_matching_result,
    get_result,
    list_thread_results,
)

router = APIRouter(tags=["results"])

_ERRORS = {
    401: {"model": ErrorResponse},
    403: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
}


@router.get(
    "/tasks/{request_id}",
    response_model=TaskStatusResponse,
    responses=_ERRORS,
    summary="Tra lại kết quả theo request_id",
)
def get_task(request_id: str, deps: DepsDep, principal: PrincipalDep) -> TaskStatusResponse:
    """Hệ thống xử lý **đồng bộ**: ``/agent/chat`` trả kết quả ngay trong response.

    Endpoint này để tra lại khi client mất response, và là chỗ sẵn sàng cho chế độ
    bất đồng bộ (khi đó ``status`` sẽ có thêm ``queued``/``running``).
    """
    results = deps.results.list_for_request(request_id=request_id, principal=principal)
    if not results:
        return TaskStatusResponse(
            request_id=request_id,
            status="not_found",
            note=(
                "không có kết quả nào còn hiệu lực cho request_id này: có thể request "
                "chưa từng tồn tại, đã hết hạn lưu trữ, hoặc thuộc người dùng khác"
            ),
        )
    return TaskStatusResponse(
        request_id=request_id,
        status="succeeded",
        thread_id=results[0].thread_id,
        results=[
            {
                "result_id": r.result_id,
                "kind": r.kind,
                "created_at": r.created_at,
                "expires_at": r.expires_at,
                "payload": r.payload,
            }
            for r in results
        ],
        note="hệ thống xử lý đồng bộ; đây là bản tra lại của kết quả đã lưu",
    )


@router.get(
    "/results/{result_id}",
    responses=_ERRORS,
    summary="Đọc một kết quả detection hoặc search đã lưu",
)
def read_result(result_id: str, deps: DepsDep, principal: PrincipalDep) -> dict:
    return get_result(result_id, deps=deps, principal=principal)


@router.get(
    "/results/{result_id}/explain",
    response_model=ExplainResponse,
    responses=_ERRORS,
    summary="Giải thích vì sao kết quả search ra trạng thái như vậy",
)
def explain_result(
    result_id: str,
    deps: DepsDep,
    principal: PrincipalDep,
    face_index: int | None = Query(
        default=None, description="chỉ giải thích một khuôn mặt cụ thể"
    ),
) -> ExplainResponse:
    """Trả lời "vì sao không tìm thấy?" bằng số đã có, không tính lại.

    Bao gồm ``what_would_change_it``: cần thêm bao nhiêu similarity nữa để vượt
    ngưỡng - hữu ích hơn nhiều so với một chữ "no_match".
    """
    envelope = get_result(result_id, deps=deps, principal=principal)
    explained = explain_matching_result(envelope["payload"], face_index=face_index)
    return ExplainResponse(result_id=result_id, **explained)


@router.get(
    "/threads/{thread_id}/results",
    response_model=ThreadResultsResponse,
    responses=_ERRORS,
    summary="Liệt kê kết quả còn hiệu lực của một thread",
)
def read_thread_results(
    thread_id: str,
    deps: DepsDep,
    principal: PrincipalDep,
    limit: int = Query(default=20, ge=1, le=100),
) -> ThreadResultsResponse:
    return ThreadResultsResponse(
        thread_id=thread_id,
        results=list_thread_results(
            thread_id, deps=deps, principal=principal, limit=limit
        ),
    )
