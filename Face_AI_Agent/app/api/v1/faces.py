"""Endpoint ``/api/v1/faces/*`` - đường đi trực tiếp, không qua agent.

Vì sao có cả đường này bên cạnh ``/agent/chat``: phần lớn tích hợp máy-với-máy
không cần ngôn ngữ tự nhiên. Chúng cần "cho ảnh, trả box" và "cho ảnh, trả top-k"
với response ổn định. Đưa những lời gọi đó qua LLM là thêm độ trễ, thêm chi phí và
thêm một nguồn không xác định, mà không đổi lấy được gì.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.deps import DepsDep, RateLimitedPrincipalDep
from app.logging_utils import new_request_id
from app.schemas.common import ErrorResponse
from app.schemas.face import (
    DeleteFacesRequest,
    DeleteFacesResponse,
    DetectRequest,
    DetectResponse,
    RegisterRequest,
    RegisterResponse,
    SearchRequest,
    SearchResponse,
)
from app.security import ROLE_DELETE, ROLE_REGISTER, ROLE_SEARCH
from app.tools.detection import DetectFacesInput, detect_faces_tool
from app.tools.search import (
    RegisterFaceInput,
    SearchFacesInput,
    register_face_tool,
    search_faces_tool,
)

router = APIRouter(prefix="/faces", tags=["faces"])

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    401: {"model": ErrorResponse},
    403: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    429: {"model": ErrorResponse},
    502: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


@router.post(
    "/detect",
    response_model=DetectResponse,
    responses=_ERROR_RESPONSES,
    summary="Phát hiện khuôn mặt trong một hoặc nhiều ảnh",
)
def detect_faces_endpoint(
    body: DetectRequest,
    deps: DepsDep,
    principal: RateLimitedPrincipalDep,
) -> DetectResponse:
    """Trả về bounding box, landmark 5 điểm và kích thước của từng khuôn mặt.

    **Không** trả confidence: backend không cung cấp. Xem ``confidence_note`` trong
    response. **Không** trả embedding: vector sinh trắc học không ra ngoài qua
    response thông thường.
    """
    principal.require_role(ROLE_SEARCH)
    result = detect_faces_tool(
        DetectFacesInput(images_base64=body.images_base64, store_result=body.store_result),
        deps=deps,
        principal=principal,
        thread_id=body.thread_id,
        request_id=new_request_id(),
    )
    return DetectResponse.model_validate(result)


@router.post(
    "/search",
    response_model=SearchResponse,
    responses=_ERROR_RESPONSES,
    summary="Tìm ứng viên khớp với (các) khuôn mặt trong ảnh truy vấn",
)
def search_faces_endpoint(
    body: SearchRequest,
    deps: DepsDep,
    principal: RateLimitedPrincipalDep,
) -> SearchResponse:
    """Tìm kiếm 1:N trong collection mà principal được phép truy cập.

    ``match_status`` có năm giá trị, trong đó ``ambiguous`` và ``needs_review`` nghĩa
    là **không được kết luận** - phải để người xem lại. ``similarity_score`` là cosine
    similarity, không phải xác suất danh tính.
    """
    request_id = new_request_id()
    result = search_faces_tool(
        SearchFacesInput(
            image_base64=body.image_base64,
            top_k=body.top_k,
            face_indices=body.face_indices,
            metadata_filter=body.metadata_filter,
            collection=body.collection,
            store_result=body.store_result,
        ),
        deps=deps,
        principal=principal,
        thread_id=body.thread_id,
        request_id=request_id,
    )
    result["execution_metadata"] = {
        "embedding_space": deps.embedding.embedding_space,
        "vector_backend": deps.repository.health().get("backend"),
        "threshold_calibrated": deps.settings.matching.calibrated,
    }
    return SearchResponse.model_validate(result)


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    responses=_ERROR_RESPONSES,
    summary="Đăng ký một khuôn mặt vào collection",
)
def register_face_endpoint(
    body: RegisterRequest,
    deps: DepsDep,
    principal: RateLimitedPrincipalDep,
) -> RegisterResponse:
    """Thêm một khuôn mặt vào index dưới một ``subject_id``.

    Ảnh có nhiều khuôn mặt thì **phải** chỉ rõ ``face_index``: hệ thống không tự
    chọn, vì đăng ký sai người làm bẩn index và rất khó phát hiện về sau.
    """
    principal.require_role(ROLE_REGISTER)
    result = register_face_tool(
        RegisterFaceInput(
            image_base64=body.image_base64,
            subject_id=body.subject_id,
            collection=body.collection,
            face_index=body.face_index,
            metadata=body.metadata,
            consent_ref=body.consent_ref,
            visibility=body.visibility,
        ),
        deps=deps,
        principal=principal,
        request_id=new_request_id(),
    )
    return RegisterResponse.model_validate(result)


@router.post(
    "/delete",
    response_model=DeleteFacesResponse,
    responses=_ERROR_RESPONSES,
    summary="Xoá/thu hồi khuôn mặt khỏi collection",
)
def delete_faces_endpoint(
    body: DeleteFacesRequest,
    deps: DepsDep,
    principal: RateLimitedPrincipalDep,
) -> DeleteFacesResponse:
    """Phục vụ chính sách lưu trữ và quyền xoá dữ liệu của chủ thể."""
    principal.require_role(ROLE_DELETE)
    deleted = deps.search.delete_faces(body.face_ids, principal=principal)
    return DeleteFacesResponse(request_id=new_request_id(), deleted=deleted)
