"""Validate & chuẩn hoá ảnh đầu vào, trước khi bất kỳ model nào được gọi.

Nguyên tắc bảo mật áp dụng ở đây:

* **Không tin metadata client gửi.** MIME type được xác định bằng magic bytes của
  chính nội dung, không phải bằng header ``Content-Type`` hay phần mở rộng tên file.
* **Chặn decompression bomb.** Một JPEG 40 KB có thể khai báo 60000x60000 pixel.
  Ta đọc header để biết kích thước *trước khi* decode toàn bộ pixel.
* **Không log nội dung ảnh.** ``ValidatedImage`` cố tình không có ``__repr__`` chứa
  bytes, và ``sha256`` là thứ duy nhất được phép xuất hiện trong log.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from PIL import Image, UnidentifiedImageError

from app.config import RequestLimitsConfig
from app.errors import (
    ImageDecodeFailed,
    ImageTooLarge,
    UnsupportedImageType,
    ValidationFailed,
)
from app.vision.base import LANDMARK_NAMES, DetectedFace, FaceBox, Landmark

# Magic bytes -> MIME. Chỉ nhận những định dạng Pillow đọc chắc chắn.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"BM", "image/bmp"),
)


def sniff_mime(data: bytes) -> str | None:
    """Đoán MIME từ nội dung. Trả None nếu không nhận ra."""
    for magic, mime in _MAGIC:
        if data.startswith(magic):
            return mime
    # WEBP: "RIFF" .... "WEBP"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


@dataclass(slots=True)
class ValidatedImage:
    """Ảnh đã qua kiểm tra, sẵn sàng đưa xuống backend.

    ``raw`` được giữ trong bộ nhớ của request; không ghi ra đĩa trừ khi
    ``storage.persist_query_images`` được bật tường minh.
    """

    raw: bytes = field(repr=False)
    mime: str
    width: int
    height: int
    sha256: str
    source: str = "inline_base64"

    @property
    def size_bytes(self) -> int:
        return len(self.raw)

    def to_base64(self) -> str:
        """Base64 không có prefix data-URL - đúng dạng backend imintv3 mong đợi."""
        return base64.b64encode(self.raw).decode("ascii")

    def log_fields(self) -> dict[str, object]:
        """Chỉ những field KHÔNG tiết lộ nội dung ảnh."""
        return {
            "image_sha256": self.sha256[:16],
            "image_mime": self.mime,
            "image_width": self.width,
            "image_height": self.height,
            "image_bytes": self.size_bytes,
        }


def decode_base64_image(payload: str) -> bytes:
    """Decode base64 nghiêm ngặt, có hỗ trợ data-URL prefix."""
    if not isinstance(payload, str) or not payload.strip():
        raise ValidationFailed("base64_image rỗng")

    text = payload.strip()
    if text.startswith("data:"):
        _, _, after = text.partition(",")
        if not after:
            raise ImageDecodeFailed("data-URL không có phần payload")
        text = after
    text = "".join(text.split())  # bỏ newline/space do copy-paste

    try:
        # validate=True để base64 rác không bị bỏ qua im lặng.
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageDecodeFailed(f"base64 không hợp lệ: {exc}") from exc


def validate_image_bytes(
    data: bytes,
    limits: RequestLimitsConfig,
    *,
    source: str = "inline_base64",
) -> ValidatedImage:
    """Kiểm tra toàn bộ rồi trả ``ValidatedImage``. Raise ``ValidationFailed`` nếu không đạt."""
    if not data:
        raise ImageDecodeFailed("ảnh rỗng sau khi decode")

    if len(data) > limits.max_image_bytes:
        raise ImageTooLarge(
            f"ảnh {len(data)} bytes vượt giới hạn {limits.max_image_bytes} bytes",
            details={"size_bytes": len(data), "max_bytes": limits.max_image_bytes},
        )

    mime = sniff_mime(data)
    if mime is None:
        raise UnsupportedImageType(
            "không nhận ra định dạng ảnh từ nội dung file",
            details={"allowed": limits.allowed_mime_types},
        )
    if mime not in limits.allowed_mime_types:
        raise UnsupportedImageType(
            f"định dạng {mime} không được phép",
            details={"detected_mime": mime, "allowed": limits.allowed_mime_types},
        )

    # Đọc header trước: .size có sẵn sau open() mà chưa decode pixel.
    try:
        with Image.open(io.BytesIO(data)) as probe:
            width, height = probe.size
            declared_pixels = width * height
            if declared_pixels > limits.max_image_pixels:
                raise ImageTooLarge(
                    f"ảnh khai báo {width}x{height} = {declared_pixels} pixel, vượt "
                    f"giới hạn {limits.max_image_pixels}",
                    details={"width": width, "height": height},
                )
            # Chỉ sau khi biết kích thước an toàn mới decode thật để chắc file không hỏng.
            probe.load()
    except ImageTooLarge:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageDecodeFailed(f"không decode được ảnh: {exc}") from exc

    if min(width, height) < limits.min_image_side_px:
        raise ValidationFailed(
            f"ảnh {width}x{height} nhỏ hơn cạnh tối thiểu {limits.min_image_side_px}px",
            details={"width": width, "height": height},
        )

    return ValidatedImage(
        raw=data,
        mime=mime,
        width=width,
        height=height,
        sha256=hashlib.sha256(data).hexdigest(),
        source=source,
    )


def validate_base64_image(
    payload: str,
    limits: RequestLimitsConfig,
    *,
    source: str = "inline_base64",
) -> ValidatedImage:
    return validate_image_bytes(decode_base64_image(payload), limits, source=source)


def parse_landmarks(values: Sequence[float] | None) -> list[Landmark]:
    """10 float phẳng -> 5 điểm có tên.

    Nếu backend đổi số điểm, ta ghép được bao nhiêu cặp thì ghép, phần thừa đặt tên
    ``point_N`` thay vì raise - landmark là thông tin phụ, không nên làm sập cả request.
    """
    if not values:
        return []
    out: list[Landmark] = []
    pairs = len(values) // 2
    for i in range(pairs):
        name = LANDMARK_NAMES[i] if i < len(LANDMARK_NAMES) else f"point_{i}"
        out.append(Landmark(name=name, x=float(values[2 * i]), y=float(values[2 * i + 1])))
    return out


def order_faces(faces: Iterable[DetectedFace]) -> list[DetectedFace]:
    """Sắp thứ tự ổn định và gán lại ``face_index``.

    Quy tắc: diện tích box giảm dần, tie-break theo (x1, y1). Vì sao cần:
    backend trả mặt theo thứ tự nội bộ của detector - đo được trên ảnh 3 mặt là
    thứ tự không theo vị trí. Người dùng thì nói "khuôn mặt thứ hai" và mong nó
    chỉ đúng một khuôn mặt, ổn định giữa các lượt hội thoại.
    """
    ordered = sorted(
        faces,
        key=lambda f: (-f.box.area, round(f.box.x1, 3), round(f.box.y1, 3)),
    )
    for idx, face in enumerate(ordered):
        face.face_index = idx
    return ordered


def mark_small_faces(faces: Iterable[DetectedFace], min_side_px: int) -> list[str]:
    """Đánh dấu mặt quá nhỏ là không dùng được cho search.

    Trả về danh sách warning để đưa lên ``FaceAnalysis.warnings``. Không xoá mặt:
    người dùng vẫn có quyền biết là đã phát hiện ra nó.
    """
    warnings: list[str] = []
    for face in faces:
        if face.box.min_side < min_side_px:
            face.rejected_reason = (
                f"face_too_small: cạnh nhỏ nhất {face.box.min_side:.0f}px < {min_side_px}px"
            )
            warnings.append(
                f"face #{face.face_index} bị loại khỏi search: quá nhỏ "
                f"({face.box.width:.0f}x{face.box.height:.0f}px)"
            )
    return warnings


def box_from_sequence(values: Sequence[float]) -> FaceBox:
    return FaceBox.from_sequence(values)
