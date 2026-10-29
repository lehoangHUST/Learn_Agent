"""Schema cho các endpoint ``/faces/*``."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.common import FacePublic


class DetectRequest(BaseModel):
    images_base64: list[str] = Field(
        min_length=1,
        max_length=16,
        description="danh sách ảnh base64. Giới hạn thực tế lấy theo limits trong config.",
    )
    store_result: bool = Field(
        default=True,
        description="lưu kết quả + embedding để lượt sau search được mà không gửi lại ảnh",
    )
    thread_id: str | None = Field(
        default=None, description="gắn kết quả vào một cuộc hội thoại"
    )

    model_config = {
        "json_schema_extra": {
            "example": {"images_base64": ["<base64 ảnh JPEG>"], "store_result": True}
        }
    }


class ImageDetectionResult(BaseModel):
    image_index: int
    face_count: int
    usable_face_count: int
    faces: list[FacePublic] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    image: dict[str, Any] = Field(
        default_factory=dict, description="sha256, mime, width, height, size_bytes"
    )
    backend: str | None = None
    backend_message: str | None = None
    latency_ms: float | None = None
    stored_image_ref: str | None = None


class DetectResponse(BaseModel):
    request_id: str
    result_id: str | None = Field(
        default=None, description="null nếu store_result=false"
    )
    image_count: int
    total_face_count: int
    usable_face_count: int
    detector_confidence_available: bool = Field(
        description="false với backend imintv3 - xem confidence_note"
    )
    confidence_note: str
    embedding_space: str
    embedding_dim: int
    images: list[ImageDetectionResult]
    faces: list[FacePublic] = Field(
        description="tất cả khuôn mặt của mọi ảnh, đánh số theo global_face_index"
    )
    warnings: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    image_base64: str = Field(description="ảnh truy vấn, base64")
    top_k: int | None = Field(default=None, ge=1, le=100)
    face_indices: list[int] | None = Field(
        default=None, description="chỉ truy vấn các khuôn mặt này (0-based); null = tất cả"
    )
    metadata_filter: dict[str, Any] | None = Field(
        default=None,
        description="chỉ xét record khớp metadata. Giá trị dạng list nghĩa là 'thuộc list'.",
    )
    collection: str | None = None
    store_result: bool = True
    thread_id: str | None = None


class CandidateOut(BaseModel):
    rank: int
    face_id: str
    subject_id: str
    similarity_score: float = Field(
        description="cosine similarity trong [-1, 1]. KHÔNG phải xác suất danh tính."
    )
    collection: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    metadata_visible: bool = Field(
        description="false nếu principal không có quyền face:read_metadata"
    )


class ThresholdOut(BaseModel):
    metric: str
    match_threshold: float
    review_band: float
    ambiguous_margin: float
    calibrated: bool = Field(
        description="false = ngưỡng chưa hiệu chuẩn, kết quả chỉ là ứng viên cần xác nhận"
    )


class FaceMatchOut(BaseModel):
    face_index: int
    match_status: Literal["match", "ambiguous", "needs_review", "no_match", "no_candidates"]
    best_score: float | None = None
    runner_up_score: float | None = None
    margin: float | None = Field(default=None, description="best_score - runner_up_score")
    threshold: float
    reason: str = Field(description="căn cứ dẫn tới match_status, bằng số")
    candidates: list[CandidateOut] = Field(default_factory=list)


class SearchResponse(BaseModel):
    request_id: str
    result_id: str | None = None
    status: Literal["ok"] = "ok"
    query_face_count: int
    collection: str
    threshold: ThresholdOut
    top_k: int
    index: dict[str, int] = Field(description="total_faces, total_subjects")
    results: list[FaceMatchOut]
    skipped_faces: list[dict[str, Any]] = Field(default_factory=list)
    detection: dict[str, Any] | None = Field(
        default=None, description="kết quả detect trên ảnh truy vấn"
    )
    warnings: list[str] = Field(default_factory=list)
    score_semantics: str
    execution_metadata: dict[str, Any] = Field(default_factory=dict)


class RegisterRequest(BaseModel):
    image_base64: str
    subject_id: str = Field(min_length=1)
    collection: str | None = None
    face_index: int | None = Field(
        default=None,
        description="bắt buộc khi ảnh có nhiều khuôn mặt - hệ thống không tự chọn",
    )
    metadata: dict[str, Any] | None = None
    consent_ref: str | None = Field(
        default=None,
        description=(
            "tham chiếu tới bằng chứng đồng ý. Thiếu field này vẫn đăng ký được nhưng "
            "sẽ bị ghi WARNING vào log - đăng ký dữ liệu sinh trắc học không có cơ sở "
            "đồng ý phải nhìn thấy được."
        ),
    )
    visibility: Literal["collection", "private"] = "collection"


class RegisterResponse(BaseModel):
    request_id: str
    subject_id: str
    collection: str
    registered_face_ids: list[str]
    face_index: int
    embedding_space: str
    has_consent_ref: bool
    index_stats: dict[str, Any]


class DeleteFacesRequest(BaseModel):
    face_ids: list[str] = Field(min_length=1)


class DeleteFacesResponse(BaseModel):
    request_id: str
    deleted: int
