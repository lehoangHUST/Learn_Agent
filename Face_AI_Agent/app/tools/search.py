"""Tool: tìm kiếm và đăng ký khuôn mặt (đường đi trực tiếp, không qua agent).

Dùng bởi endpoint ``/faces/search``, ``/faces/register`` và bởi
``scripts/register_face.py``. Xem docstring của ``app/tools/detection.py`` về việc
vì sao các tool này không được đăng ký cho LLM gọi.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from app.dependencies import AgentDeps
from app.errors import ValidationFailed
from app.logging_utils import new_request_id
from app.repositories.result_store import KIND_SEARCH
from app.security import Principal

logger = logging.getLogger(__name__)


class SearchFacesInput(BaseModel):
    image_base64: str = Field(description="ảnh truy vấn, base64")
    top_k: int | None = Field(default=None, ge=1, le=100)
    face_indices: list[int] | None = Field(
        default=None,
        description="chỉ truy vấn những khuôn mặt này (0-based); null = tất cả",
    )
    metadata_filter: dict[str, Any] | None = Field(
        default=None,
        description="chỉ xét record có metadata khớp; giá trị là list nghĩa là 'thuộc list'",
    )
    collection: str | None = None
    store_result: bool = True


class RegisterFaceInput(BaseModel):
    image_base64: str
    subject_id: str = Field(min_length=1, description="định danh người trong tập đăng ký")
    collection: str | None = None
    face_index: int | None = Field(
        default=None,
        description="ảnh nhiều mặt thì chỉ rõ mặt nào; null = chỉ cho phép khi ảnh có đúng 1 mặt",
    )
    metadata: dict[str, Any] | None = None
    consent_ref: str | None = Field(
        default=None,
        description="tham chiếu tới bằng chứng đồng ý thu thập dữ liệu sinh trắc học",
    )
    visibility: str = Field(default="collection", pattern="^(collection|private)$")


def search_faces_tool(
    payload: SearchFacesInput,
    *,
    deps: AgentDeps,
    principal: Principal,
    thread_id: str | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Tìm ứng viên cho từng khuôn mặt trong ảnh truy vấn."""
    request_id = request_id or new_request_id()
    collection = payload.collection or deps.settings.vector_store.collection

    image = deps.detection.validate_many([payload.image_base64])[0]
    outcome = deps.detection.detect(image)
    deps.embedding.ensure_embeddings(outcome.analysis, image.raw)

    faces, skipped = deps.embedding.select_faces(outcome.analysis, payload.face_indices)
    batch = deps.embedding.build_matrix(faces)
    batch.skipped.extend(skipped)

    search_outcome = deps.search.search(
        batch,
        principal=principal,
        collection=collection,
        top_k=payload.top_k,
        metadata_filter=payload.metadata_filter,
    )

    result = search_outcome.to_public_dict()
    result["request_id"] = request_id
    result["kind"] = KIND_SEARCH
    result["detection"] = {
        "total_face_count": outcome.analysis.face_count,
        "usable_face_count": len(outcome.analysis.usable_faces),
        "detector_confidence_available": any(
            f.confidence is not None for f in outcome.analysis.faces
        ),
        "faces": [
            f.to_public_dict(image.width, image.height) for f in outcome.analysis.faces
        ],
    }

    if payload.store_result:
        result["result_id"] = deps.results.save(
            kind=KIND_SEARCH,
            owner_id=principal.user_id,
            payload=result,
            thread_id=thread_id,
            request_id=request_id,
            # Lưu embedding của ảnh truy vấn để lượt sau hỏi tiếp không phải gửi lại ảnh.
            embeddings={
                face.face_index: face.embedding
                for face in faces
                if face.embedding is not None
            }
            or None,
            embedding_space=deps.embedding.embedding_space if faces else None,
        )

    logger.info(
        "tool_search_faces",
        extra={
            "request_id": request_id,
            "collection": collection,
            "query_faces": search_outcome.query_face_count,
            **principal.log_fields(),
        },
    )
    return result


def register_face_tool(
    payload: RegisterFaceInput,
    *,
    deps: AgentDeps,
    principal: Principal,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Đăng ký một khuôn mặt vào index dưới một ``subject_id``."""
    request_id = request_id or new_request_id()
    collection = payload.collection or deps.settings.vector_store.collection

    image = deps.detection.validate_many([payload.image_base64])[0]
    outcome = deps.detection.detect(image)
    deps.embedding.ensure_embeddings(outcome.analysis, image.raw)

    usable = outcome.analysis.usable_faces
    if not usable:
        raise ValidationFailed(
            "không có khuôn mặt nào dùng được trong ảnh để đăng ký",
            details={
                "face_count": outcome.analysis.face_count,
                "warnings": outcome.analysis.warnings,
            },
        )

    if payload.face_index is None:
        if len(usable) > 1:
            # Không tự chọn: đăng ký sai người vào index là loại lỗi rất khó phát hiện
            # về sau và làm bẩn toàn bộ kết quả search.
            raise ValidationFailed(
                f"ảnh có {len(usable)} khuôn mặt dùng được; hãy chỉ rõ face_index",
                details={"available_face_indices": [f.face_index for f in usable]},
            )
        chosen = [usable[0]]
    else:
        chosen = [f for f in usable if f.face_index == payload.face_index]
        if not chosen:
            raise ValidationFailed(
                f"không có khuôn mặt dùng được ở index {payload.face_index}",
                details={"available_face_indices": [f.face_index for f in usable]},
            )

    face_ids = deps.search.register_faces(
        chosen,
        subject_id=payload.subject_id,
        principal=principal,
        collection=collection,
        embedding_space=deps.embedding.embedding_space,
        metadata={**(payload.metadata or {}), "image_sha256": image.sha256},
        consent_ref=payload.consent_ref,
        visibility=payload.visibility,
    )

    return {
        "request_id": request_id,
        "subject_id": payload.subject_id,
        "collection": collection,
        "registered_face_ids": face_ids,
        "face_index": chosen[0].face_index,
        "embedding_space": deps.embedding.embedding_space,
        "has_consent_ref": payload.consent_ref is not None,
        "index_stats": deps.search.stats(collection),
    }
