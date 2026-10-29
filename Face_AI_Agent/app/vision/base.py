"""Kiểu dữ liệu và interface của tầng Computer Vision.

Tầng này KHÔNG biết gì về LangGraph, FastAPI hay FAISS. Nó chỉ nhận ảnh đã được
validate và trả về mô tả các khuôn mặt tìm thấy.

Về việc tách interface
----------------------
Lý thuyết chuẩn là tách ``BaseFaceDetector`` và ``BaseFaceEmbedder`` thành hai
thành phần độc lập. Backend thực tế (imintv3) lại trả box + landmark + embedding
**trong cùng một lời gọi HTTP**, nên ``analyze()`` mới là primitive thật; detect và
embed chỉ là hai góc nhìn vào cùng kết quả đó.

Thiết kế ở đây giữ cả ba: ``BaseFaceEngine.analyze`` là primitive, còn
``BaseFaceDetector`` / ``BaseFaceEmbedder`` là hai Protocol hẹp hơn để service tầng
trên chỉ phụ thuộc vào đúng phần nó cần. Một adapter có thể hiện thực cả ba (như
``ImintV3FaceEngine``), hoặc hai adapter riêng nếu sau này đổi sang
SCRFD + ArcFace chạy local - tầng service không phải sửa dòng nào.

Cái giá của việc gộp: gọi detect rồi gọi embed trên cùng một ảnh sẽ là 2 lời gọi
HTTP nếu làm ngây thơ. ``ImintV3FaceEngine`` giải quyết bằng memo theo nội dung ảnh.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

# Layout 5 điểm của imintv3: 10 float = (x, y) * 5 theo thứ tự dưới đây.
# Xác nhận bằng giá trị thật: hai điểm đầu có y ~ 52-53 (mắt), điểm 3 y ~ 67 (mũi),
# hai điểm cuối y ~ 78-79 (miệng).
LANDMARK_NAMES: tuple[str, ...] = (
    "left_eye",
    "right_eye",
    "nose",
    "mouth_left",
    "mouth_right",
)


@dataclass(frozen=True, slots=True)
class FaceBox:
    """Bounding box theo pixel, hệ toạ độ gốc trên-trái, dạng [x1, y1, x2, y2]."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def min_side(self) -> float:
        return min(self.width, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> FaceBox:
        if len(values) < 4:
            raise ValueError(f"box cần ít nhất 4 giá trị, nhận được {len(values)}")
        x1, y1, x2, y2 = (float(v) for v in values[:4])
        # Một số detector trả toạ độ ngược; chuẩn hoá để width/height không âm.
        return cls(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


@dataclass(frozen=True, slots=True)
class Landmark:
    name: str
    x: float
    y: float


@dataclass(slots=True)
class DetectedFace:
    """Một khuôn mặt trong một ảnh.

    ``face_index`` được gán sau khi sort ổn định (diện tích giảm dần, tie-break theo
    toạ độ) - xem ``preprocessing.order_faces``. Điều này quan trọng vì backend trả
    các mặt theo thứ tự nội bộ của detector, không ổn định giữa các ảnh; mà người
    dùng lại nói "khuôn mặt thứ hai" và mong nó cố định giữa các lượt hội thoại.
    """

    face_index: int
    box: FaceBox
    landmarks: list[Landmark] = field(default_factory=list)

    # ---- confidence ----
    # Backend imintv3 KHÔNG trả confidence (boxes chỉ có 4 số: x1,y1,x2,y2).
    # Giữ None và nói rõ qua confidence_source. Không bao giờ điền số tự nghĩ ra:
    # một "confidence" bịa đặt còn tệ hơn là không có confidence.
    confidence: float | None = None
    confidence_source: str = "unavailable"

    embedding: np.ndarray | None = None
    embedding_space: str | None = None

    # Lý do mặt này bị loại khỏi bước search (ví dụ quá nhỏ). None = dùng được.
    rejected_reason: str | None = None

    @property
    def has_embedding(self) -> bool:
        return self.embedding is not None

    @property
    def is_usable(self) -> bool:
        return self.rejected_reason is None

    def size_facts(self, image_width: int | None, image_height: int | None) -> dict[str, Any]:
        """Các dữ kiện *khách quan* về kích thước.

        Đây KHÔNG phải confidence và không được trình bày như confidence. Nó chỉ
        giúp trả lời được những câu như "mặt nào to nhất trong ảnh".
        """
        facts: dict[str, Any] = {
            "width_px": round(self.box.width, 1),
            "height_px": round(self.box.height, 1),
            "area_px": round(self.box.area, 1),
        }
        if image_width and image_height:
            total = float(image_width * image_height)
            facts["relative_area"] = round(self.box.area / total, 5) if total > 0 else None
        return facts

    def to_public_dict(self, image_width: int | None = None, image_height: int | None = None) -> dict[str, Any]:
        """Dạng an toàn để trả ra API / đưa vào prompt: KHÔNG chứa embedding."""
        return {
            "face_index": self.face_index,
            "box": [round(v, 2) for v in self.box.as_list()],
            "landmarks": [{"name": lm.name, "x": round(lm.x, 2), "y": round(lm.y, 2)} for lm in self.landmarks],
            "confidence": self.confidence,
            "confidence_source": self.confidence_source,
            "size": self.size_facts(image_width, image_height),
            "has_embedding": self.has_embedding,
            "embedding_space": self.embedding_space,
            "usable_for_search": self.is_usable,
            "rejected_reason": self.rejected_reason,
        }


@dataclass(slots=True)
class FaceAnalysis:
    """Kết quả phân tích một ảnh."""

    faces: list[DetectedFace]
    image_width: int | None = None
    image_height: int | None = None
    backend: str = "unknown"
    backend_message: str | None = None
    latency_ms: float | None = None
    # Những chuyện đáng nói nhưng không phải lỗi: mặt bị loại vì nhỏ, backend báo
    # "Error" khi không có mặt nào, ...
    warnings: list[str] = field(default_factory=list)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    @property
    def usable_faces(self) -> list[DetectedFace]:
        return [f for f in self.faces if f.is_usable]

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "face_count": self.face_count,
            "usable_face_count": len(self.usable_faces),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "backend": self.backend,
            "detector_confidence_available": any(f.confidence is not None for f in self.faces),
            "faces": [f.to_public_dict(self.image_width, self.image_height) for f in self.faces],
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------- #
# Interfaces
# --------------------------------------------------------------------------- #
@runtime_checkable
class BaseFaceEngine(Protocol):
    """Primitive: một ảnh -> các khuôn mặt (box + landmark + embedding)."""

    name: str

    def analyze(self, image_bytes: bytes) -> FaceAnalysis:
        """Phân tích ảnh. Raise ``VisionBackendError``/``ValidationFailed`` khi thất bại.

        Ảnh không có mặt nào KHÔNG phải lỗi: trả ``FaceAnalysis`` với ``faces == []``.
        """
        ...

    def health(self) -> dict[str, Any]:
        """Thông tin trạng thái backend cho endpoint /health."""
        ...


@runtime_checkable
class BaseFaceDetector(Protocol):
    """Góc nhìn 'chỉ cần biết mặt ở đâu'."""

    def detect(self, image_bytes: bytes) -> FaceAnalysis: ...


@runtime_checkable
class BaseFaceEmbedder(Protocol):
    """Góc nhìn 'cần vector để so khớp'."""

    @property
    def embedding_dim(self) -> int: ...

    @property
    def embedding_space(self) -> str: ...

    def embed(self, image_bytes: bytes) -> FaceAnalysis: ...
