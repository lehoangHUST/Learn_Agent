"""Schema dùng chung cho API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str = Field(description="mã lỗi ổn định, dùng để client phân loại")
    message: str
    retryable: bool = Field(description="gọi lại y nguyên có cơ hội thành công hay không")
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Mọi lỗi của API đều có hình dạng này."""

    request_id: str
    error: ErrorBody

    model_config = {
        "json_schema_extra": {
            "example": {
                "request_id": "req_8f21c0a3bb1e4d57",
                "error": {
                    "code": "image_too_large",
                    "message": "ảnh 12582912 bytes vượt giới hạn 8388608 bytes",
                    "retryable": False,
                    "details": {"size_bytes": 12582912, "max_bytes": 8388608},
                },
            }
        }
    }


class ComponentHealth(BaseModel):
    name: str
    ok: bool
    detail: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    app_name: str
    version: str
    environment: str
    components: list[ComponentHealth]
    # Nói thẳng ngay ở /health: ngưỡng đã hiệu chuẩn chưa. Đây là thông tin vận hành
    # quan trọng hơn cả uptime - một hệ thống "ok" với ngưỡng chưa hiệu chuẩn vẫn
    # không nên dùng để quyết định danh tính.
    matching_threshold_calibrated: bool
    warnings: list[str] = Field(default_factory=list)


class SizeFacts(BaseModel):
    width_px: float | None = None
    height_px: float | None = None
    area_px: float | None = None
    relative_area: float | None = Field(
        default=None, description="diện tích box / diện tích ảnh"
    )


class LandmarkPoint(BaseModel):
    name: str = Field(description="left_eye | right_eye | nose | mouth_left | mouth_right")
    x: float
    y: float


class FacePublic(BaseModel):
    """Một khuôn mặt, dạng trả ra ngoài. Không bao giờ chứa embedding."""

    face_index: int = Field(description="chỉ số trong ảnh của nó")
    global_face_index: int | None = Field(
        default=None, description="chỉ số chạy suốt các ảnh trong cùng request"
    )
    image_index: int | None = None
    box: list[float] = Field(description="[x1, y1, x2, y2] theo pixel")
    landmarks: list[LandmarkPoint] = Field(default_factory=list)
    confidence: float | None = Field(
        default=None,
        description=(
            "LUÔN null với backend imintv3: detector không trả confidence. "
            "Không suy ra confidence từ kích thước box."
        ),
    )
    confidence_source: str = "unavailable_from_backend"
    size: SizeFacts = Field(default_factory=SizeFacts)
    has_embedding: bool = False
    embedding_space: str | None = None
    usable_for_search: bool = True
    rejected_reason: str | None = None
