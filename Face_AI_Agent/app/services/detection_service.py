"""Face Detection Service - biên giới giữa "ảnh thô từ người dùng" và tầng vision.

Trách nhiệm:

1. Validate ảnh (kích thước, MIME theo magic bytes, pixel bomb) - uỷ quyền cho
   ``app.vision.preprocessing``.
2. Gọi detector.
3. Bù những thông tin mà backend không trả nhưng ta biết chắc: kích thước ảnh gốc.
   Nhờ đó mới tính được ``relative_area`` của từng khuôn mặt.

Service này **không** quyết định chính sách nghiệp vụ (có search hay không, trả lời
thế nào) - đó là việc của graph. Nó cũng không biết gì về HTTP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.config import RequestLimitsConfig
from app.errors import ValidationFailed
from app.logging_utils import timed
from app.vision.base import BaseFaceDetector, FaceAnalysis
from app.vision.preprocessing import (
    ValidatedImage,
    validate_base64_image,
    validate_image_bytes,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DetectionOutcome:
    """Kết quả detect trên một ảnh, kèm metadata ảnh đã được xác thực."""

    image: ValidatedImage
    analysis: FaceAnalysis

    def to_public_dict(self) -> dict[str, Any]:
        """Payload an toàn: không có embedding, không có base64."""
        data = self.analysis.to_public_dict()
        data["image"] = {
            "sha256": self.image.sha256,
            "mime": self.image.mime,
            "width": self.image.width,
            "height": self.image.height,
            "size_bytes": self.image.size_bytes,
            "source": self.image.source,
        }
        return data


class FaceDetectionService:
    def __init__(self, detector: BaseFaceDetector, limits: RequestLimitsConfig) -> None:
        self._detector = detector
        self._limits = limits

    def detect_from_base64(self, payload: str, *, source: str = "inline_base64") -> DetectionOutcome:
        image = validate_base64_image(payload, self._limits, source=source)
        return self.detect(image)

    def detect_from_bytes(self, data: bytes, *, source: str = "bytes") -> DetectionOutcome:
        image = validate_image_bytes(data, self._limits, source=source)
        return self.detect(image)

    def detect(self, image: ValidatedImage) -> DetectionOutcome:
        with timed(logger, "face_detect", **image.log_fields()) as extra:
            analysis = self._detector.detect(image.raw)
            # Backend không trả kích thước ảnh; ta biết chắc từ bước validate.
            analysis.image_width = image.width
            analysis.image_height = image.height
            extra["face_count"] = analysis.face_count
            extra["usable_face_count"] = len(analysis.usable_faces)
            extra["backend_message"] = analysis.backend_message
        return DetectionOutcome(image=image, analysis=analysis)

    def validate_many(
        self, payloads: list[str], *, source: str = "inline_base64"
    ) -> list[ValidatedImage]:
        """Validate cả lô trước khi gọi model.

        Fail-fast có chủ ý: nếu ảnh thứ 3 trong 5 ảnh không hợp lệ thì ta báo lỗi
        ngay chứ không chạy model cho 2 ảnh đầu rồi mới lỗi - vừa tốn GPU, vừa để
        lại kết quả nửa vời mà người dùng không gọi lại được nguyên trạng.
        """
        if not payloads:
            raise ValidationFailed("request không có ảnh nào")
        if len(payloads) > self._limits.max_images_per_request:
            raise ValidationFailed(
                f"{len(payloads)} ảnh vượt giới hạn {self._limits.max_images_per_request} ảnh/request",
                details={"count": len(payloads), "max": self._limits.max_images_per_request},
            )
        images: list[ValidatedImage] = []
        for idx, payload in enumerate(payloads):
            try:
                images.append(validate_base64_image(payload, self._limits, source=source))
            except ValidationFailed as exc:
                # Nói rõ ảnh thứ mấy sai, nhưng không echo nội dung ảnh.
                exc.details.setdefault("image_index", idx)
                raise
        return images
