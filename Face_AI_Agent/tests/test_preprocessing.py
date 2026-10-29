"""Test validate ảnh - tầng chặn đầu tiên và quan trọng nhất về bảo mật."""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from app.config import RequestLimitsConfig
from app.errors import (
    ImageDecodeFailed,
    ImageTooLarge,
    UnsupportedImageType,
    ValidationFailed,
)
from app.vision.base import DetectedFace, FaceBox
from app.vision.preprocessing import (
    decode_base64_image,
    order_faces,
    parse_landmarks,
    sniff_mime,
    validate_base64_image,
    validate_image_bytes,
)
from tests.conftest import make_image_bytes


@pytest.fixture
def limits() -> RequestLimitsConfig:
    return RequestLimitsConfig()


def test_sniff_mime_detects_formats() -> None:
    assert sniff_mime(make_image_bytes(fmt="JPEG")) == "image/jpeg"
    assert sniff_mime(make_image_bytes(fmt="PNG")) == "image/png"
    assert sniff_mime(b"not an image at all") is None


def test_mime_comes_from_content_not_from_client_claim(limits: RequestLimitsConfig) -> None:
    """Kiểm tra bằng magic bytes: một file .jpg thực chất là PNG vẫn được nhận đúng.

    Mặt khác, một file text đặt tên .jpg thì bị từ chối - đây là chỗ chặn upload
    payload không phải ảnh.
    """
    png = make_image_bytes(fmt="PNG")
    assert validate_image_bytes(png, limits).mime == "image/png"

    with pytest.raises(UnsupportedImageType):
        validate_image_bytes(b"#!/bin/sh\necho pwned\n" * 20, limits)


def test_rejects_oversized_image() -> None:
    limits = RequestLimitsConfig(max_image_bytes=100)
    with pytest.raises(ImageTooLarge) as exc:
        validate_image_bytes(make_image_bytes(), limits)
    assert exc.value.details["max_bytes"] == 100


def test_rejects_too_small_image() -> None:
    limits = RequestLimitsConfig(min_image_side_px=64)
    tiny = make_image_bytes(width=32, height=32)
    with pytest.raises(ValidationFailed):
        validate_image_bytes(tiny, limits)


def test_rejects_declared_pixel_bomb() -> None:
    """Chặn theo kích thước khai báo trong header, trước khi decode pixel."""
    limits = RequestLimitsConfig(max_image_pixels=1000)
    with pytest.raises(ImageTooLarge):
        validate_image_bytes(make_image_bytes(width=200, height=200), limits)


def test_strict_base64_decode() -> None:
    with pytest.raises(ImageDecodeFailed):
        decode_base64_image("not-a-real-base64!!!")
    with pytest.raises(ValidationFailed):
        decode_base64_image("   ")


def test_data_url_prefix_is_accepted(limits: RequestLimitsConfig) -> None:
    raw = make_image_bytes()
    payload = "data:image/jpeg;base64," + base64.b64encode(raw).decode()
    image = validate_base64_image(payload, limits)
    assert image.mime == "image/jpeg"
    assert image.size_bytes == len(raw)


def test_whitespace_in_base64_is_tolerated(limits: RequestLimitsConfig) -> None:
    """Copy-paste base64 hay bị chèn newline; không nên thành lỗi của người dùng."""
    raw = make_image_bytes()
    encoded = base64.b64encode(raw).decode()
    chunked = "\n".join(encoded[i : i + 76] for i in range(0, len(encoded), 76))
    assert validate_base64_image(chunked, limits).sha256 == validate_image_bytes(raw, limits).sha256


def test_corrupt_image_body_is_rejected(limits: RequestLimitsConfig) -> None:
    """Magic bytes đúng nhưng nội dung hỏng -> phải lỗi ở bước decode."""
    broken = make_image_bytes()[:120] + b"\x00" * 40
    with pytest.raises(ImageDecodeFailed):
        validate_image_bytes(broken, limits)


def test_validated_image_log_fields_contain_no_content(limits: RequestLimitsConfig) -> None:
    image = validate_image_bytes(make_image_bytes(), limits)
    fields = image.log_fields()
    assert set(fields) == {
        "image_sha256",
        "image_mime",
        "image_width",
        "image_height",
        "image_bytes",
    }
    assert base64.b64encode(image.raw).decode()[:40] not in repr(fields)


def test_parse_landmarks_names_five_points() -> None:
    flat = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    points = parse_landmarks(flat)
    assert [p.name for p in points] == [
        "left_eye",
        "right_eye",
        "nose",
        "mouth_left",
        "mouth_right",
    ]
    assert (points[0].x, points[0].y) == (1.0, 2.0)
    assert parse_landmarks(None) == []


def test_parse_landmarks_tolerates_extra_points() -> None:
    """Backend đổi số điểm thì không được làm sập request - landmark là thông tin phụ."""
    points = parse_landmarks(list(range(14)))
    assert len(points) == 7
    assert points[6].name == "point_6"


def test_order_faces_is_stable_and_reindexes() -> None:
    faces = [
        DetectedFace(face_index=0, box=FaceBox(0, 0, 10, 10)),      # area 100
        DetectedFace(face_index=1, box=FaceBox(0, 0, 100, 100)),    # area 10000
        DetectedFace(face_index=2, box=FaceBox(50, 0, 150, 100)),   # area 10000, x1 lớn hơn
    ]
    ordered = order_faces(faces)
    assert [f.box.area for f in ordered] == [10000, 10000, 100]
    # Tie-break theo x1 -> thứ tự xác định, chạy lại nhiều lần vẫn vậy.
    assert ordered[0].box.x1 == 0
    assert ordered[1].box.x1 == 50
    assert [f.face_index for f in ordered] == [0, 1, 2]


def test_box_coordinates_are_normalized() -> None:
    """Toạ độ đảo ngược vẫn cho width/height không âm."""
    box = FaceBox.from_sequence([100, 90, 10, 20])
    assert box.width == 90 and box.height == 70
    assert box.area == 6300


def test_box_requires_four_values() -> None:
    with pytest.raises(ValueError):
        FaceBox.from_sequence([1, 2, 3])


def test_webp_is_sniffed() -> None:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (1, 2, 3)).save(buf, format="WEBP")
    assert sniff_mime(buf.getvalue()) == "image/webp"
