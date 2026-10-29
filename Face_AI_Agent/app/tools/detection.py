"""Tool: phát hiện khuôn mặt (đường đi trực tiếp, không qua agent).

Về chuyện "tool" trong project này
----------------------------------
Yêu cầu đặt ra là phân biệt tool nào nên là hàm Python thường, tool nào nên đăng ký
cho LLM gọi, tool nào nên gọi trực tiếp từ node. Quyết định ở đây:

* **Không tool nào được đăng ký cho LLM tool-calling.** Mọi thao tác trong hệ thống
  này đều xác định rõ: validate ảnh, gọi detector, truy vấn vector. Để LLM quyết
  định "có nên gọi detector không" là đổi một nhánh ``if`` tất định thành một lời
  gọi mạng có thể lỗi và có thể chọn sai. Việc chọn bước nào chạy đã thuộc về
  ``plan_execution`` - tính bằng code từ intent.
* **Node của graph gọi service trực tiếp** (``app/services``). Đó là đường đi của
  luồng hội thoại.
* **Module ``app/tools`` là use-case layer cho đường đi KHÔNG qua agent**: REST
  endpoint ``/faces/*`` và các script CLI. Nó thêm đúng ba thứ so với việc gọi
  service: schema đầu vào có validate, kiểm tra quyền, và lắp response theo đúng
  hợp đồng API. Nhờ vậy ``scripts/register_face.py`` và endpoint HTTP dùng cùng một
  code, không lệch hành vi.

Nếu sau này muốn một agent kiểu ReAct tự chọn tool, các hàm ở đây là thứ đem bọc
``@tool`` - chúng đã có schema và đã kiểm quyền.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from app.dependencies import AgentDeps
from app.logging_utils import new_request_id
from app.repositories.result_store import KIND_DETECTION
from app.security import ROLE_SEARCH, Principal

logger = logging.getLogger(__name__)


class DetectFacesInput(BaseModel):
    """Đầu vào của ``detect_faces_tool``."""

    images_base64: list[str] = Field(
        min_length=1,
        description="danh sách ảnh base64 (không cần prefix data-URL)",
    )
    store_result: bool = Field(
        default=True,
        description=(
            "lưu kết quả (kèm embedding) vào result store để lượt sau tìm kiếm được "
            "mà không phải gửi lại ảnh"
        ),
    )


def detect_faces_tool(
    payload: DetectFacesInput,
    *,
    deps: AgentDeps,
    principal: Principal,
    thread_id: str | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Phát hiện khuôn mặt trong một hoặc nhiều ảnh.

    Trả về payload công khai: box, landmark, kích thước. **Không** trả embedding -
    vector sinh trắc học không ra khỏi hệ thống qua response thông thường.
    """
    request_id = request_id or new_request_id()
    images = deps.detection.validate_many(payload.images_base64)

    images_payload: list[dict[str, Any]] = []
    faces_flat: list[dict[str, Any]] = []
    embeddings: dict[int, Any] = {}
    warnings: list[str] = []
    global_index = 0

    for image_index, image in enumerate(images):
        outcome = deps.detection.detect(image)
        deps.embedding.ensure_embeddings(outcome.analysis, image.raw)

        per_image = outcome.to_public_dict()
        per_image["image_index"] = image_index
        per_image["faces"] = []
        for face in outcome.analysis.faces:
            public = face.to_public_dict(image.width, image.height)
            public["global_face_index"] = global_index
            public["image_index"] = image_index
            per_image["faces"].append(public)
            faces_flat.append(public)
            if face.is_usable and face.embedding is not None:
                embeddings[global_index] = face.embedding
            global_index += 1

        warnings.extend(f"ảnh #{image_index}: {w}" for w in outcome.analysis.warnings)
        images_payload.append(per_image)

    result: dict[str, Any] = {
        "request_id": request_id,
        "kind": KIND_DETECTION,
        "image_count": len(images),
        "total_face_count": len(faces_flat),
        "usable_face_count": sum(1 for f in faces_flat if f.get("usable_for_search")),
        "detector_confidence_available": any(f.get("confidence") is not None for f in faces_flat),
        "confidence_note": (
            "backend không trả confidence của detector; chỉ có kích thước bounding box, "
            "và kích thước KHÔNG phải confidence"
        ),
        "embedding_space": deps.embedding.embedding_space,
        "embedding_dim": deps.embedding.embedding_dim,
        "images": images_payload,
        "faces": faces_flat,
        "warnings": warnings,
    }

    if payload.store_result:
        result_id = deps.results.save(
            kind=KIND_DETECTION,
            owner_id=principal.user_id,
            payload=result,
            thread_id=thread_id,
            request_id=request_id,
            embeddings=embeddings or None,
            embedding_space=deps.embedding.embedding_space if embeddings else None,
        )
        result["result_id"] = result_id

    logger.info(
        "tool_detect_faces",
        extra={
            "request_id": request_id,
            "image_count": len(images),
            "face_count": len(faces_flat),
            **principal.log_fields(),
        },
    )
    return result


def require_detect_permission(principal: Principal) -> None:
    """Phát hiện khuôn mặt cũng là xử lý dữ liệu sinh trắc học -> cần quyền."""
    principal.require_role(ROLE_SEARCH)
