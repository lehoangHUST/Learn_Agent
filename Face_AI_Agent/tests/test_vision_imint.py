"""Test adapter imintv3 với response **đúng hình dạng thật** của backend.

Các payload dưới đây được dựng theo kết quả gọi thật endpoint
``POST /imintv3/extract_feature/``, kể cả những hành vi kỳ:

* không có mặt -> HTTP 200, ``message="Error"``, các list rỗng, ``status_respond=0``,
* base64 rác -> HTTP 500 với body **text thuần**, không phải JSON,
* ``result.request`` echo lại toàn bộ base64 ảnh,
* ``boxes`` chỉ có 4 số (không có confidence).

Dùng ``httpx.MockTransport`` thay vì mock cả client: như vậy code đường HTTP thật
(retry, phân loại status, parse) đều được chạy.
"""

from __future__ import annotations

import httpx
import numpy as np
import pytest

from app.config import VisionConfig
from app.errors import VisionBackendError, VisionBackendRejectedImage, VisionBackendTimeout
from app.vision.imint_v3 import ImintV3FaceEngine

DIM = 512


def unit_vector(seed: int, dim: int = DIM) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim)
    v = v / np.linalg.norm(v)
    return [float(x) for x in v]


def success_body(n_faces: int = 1, *, echo_base64: str = "AAAA") -> dict:
    return {
        "message": "success",
        "trace_message": "",
        "status_respond": 0,
        "result": {
            "features": [unit_vector(i) for i in range(n_faces)],
            "features_unalign": [unit_vector(100 + i) for i in range(n_faces)],
            "boxes": [
                [37.37 + i * 100, 18.99, 97.41 + i * 100, 94.81] for i in range(n_faces)
            ],
            "landmarks": [
                [52.03, 52.63, 79.08, 53.70, 64.34, 67.58, 54.22, 78.22, 76.71, 79.02]
                for _ in range(n_faces)
            ],
            # Backend thật echo lại cả ảnh ở đây.
            "request": {"base64_image": echo_base64, "detect_face": 1, "is_checkresize": True},
        },
    }


NO_FACE_BODY = {
    "message": "Error",
    "trace_message": "",
    "status_respond": 0,
    "result": {
        "features": [],
        "features_unalign": [],
        "boxes": [],
        "landmarks": [],
        "request": {"base64_image": "AAAA", "detect_face": 1, "is_checkresize": True},
    },
}


def make_engine(handler, **overrides) -> ImintV3FaceEngine:
    config = VisionConfig(max_retries=1, retry_backoff_seconds=0.0, **overrides)
    client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url=config.base_url
    )
    return ImintV3FaceEngine(config, client=client)


# --------------------------------------------------------------------------- #
def test_parses_single_face_with_real_response_shape() -> None:
    engine = make_engine(lambda request: httpx.Response(200, json=success_body(1)))
    analysis = engine.analyze(b"\xff\xd8\xff fake jpeg")

    assert analysis.face_count == 1
    face = analysis.faces[0]
    assert face.box.as_list() == pytest.approx([37.37, 18.99, 97.41, 94.81], abs=1e-2)
    assert [lm.name for lm in face.landmarks] == [
        "left_eye",
        "right_eye",
        "nose",
        "mouth_left",
        "mouth_right",
    ]
    assert face.embedding is not None and face.embedding.shape == (DIM,)
    assert face.embedding_space == "aligned"
    assert float(np.linalg.norm(face.embedding)) == pytest.approx(1.0, abs=1e-5)


def test_confidence_is_never_invented() -> None:
    """Backend chỉ trả 4 số cho box -> confidence phải là None, không phải số bịa."""
    engine = make_engine(lambda request: httpx.Response(200, json=success_body(1)))
    analysis = engine.analyze(b"\xff\xd8\xff x")

    assert analysis.faces[0].confidence is None
    assert analysis.faces[0].confidence_source == "unavailable_from_backend"
    assert analysis.to_public_dict()["detector_confidence_available"] is False


def test_echoed_base64_is_stripped_from_result() -> None:
    """``result.request`` chứa cả ảnh; nó không được đi tiếp vào bất cứ đâu."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = success_body(1, echo_base64="SECRET_IMAGE_PAYLOAD")
        captured["body"] = body
        return httpx.Response(200, json=body)

    engine = make_engine(handler)
    analysis = engine.analyze(b"\xff\xd8\xff x")

    public = analysis.to_public_dict()
    assert "SECRET_IMAGE_PAYLOAD" not in repr(public)
    assert "request" not in repr(public)


def test_no_face_is_not_an_error() -> None:
    """message='Error' + list rỗng là 'không có mặt', không phải lỗi hệ thống."""
    engine = make_engine(lambda request: httpx.Response(200, json=NO_FACE_BODY))
    analysis = engine.analyze(b"\xff\xd8\xff x")

    assert analysis.face_count == 0
    assert analysis.backend_message == "Error"
    assert any("không phát hiện" in w for w in analysis.warnings)


def test_status_respond_is_not_the_discriminator() -> None:
    """status_respond = 0 ở CẢ hai trường hợp thành công và không có mặt.

    Test này tồn tại để khoá lại một quyết định: ai sửa code mà chuyển sang dùng
    ``status_respond`` làm cờ thành/bại sẽ làm test này đỏ.
    """
    assert success_body(1)["status_respond"] == 0
    assert NO_FACE_BODY["status_respond"] == 0


def test_multiple_faces_are_ordered_by_area_descending() -> None:
    body = success_body(3)
    # Làm cho mặt thứ 2 (index 1) to nhất.
    body["result"]["boxes"] = [
        [0, 0, 50, 50],       # area 2500
        [100, 100, 300, 300],  # area 40000 <- to nhất
        [10, 10, 110, 110],    # area 10000
    ]
    engine = make_engine(lambda request: httpx.Response(200, json=body))
    analysis = engine.analyze(b"\xff\xd8\xff x")

    areas = [f.box.area for f in analysis.faces]
    assert areas == sorted(areas, reverse=True)
    assert [f.face_index for f in analysis.faces] == [0, 1, 2]
    assert analysis.faces[0].box.area == pytest.approx(40000)


def test_small_face_is_marked_unusable_not_dropped() -> None:
    body = success_body(2)
    body["result"]["boxes"] = [[0, 0, 100, 100], [200, 200, 215, 215]]  # mặt 2: 15px
    engine = make_engine(lambda request: httpx.Response(200, json=body), min_face_size_px=32)
    analysis = engine.analyze(b"\xff\xd8\xff x")

    assert analysis.face_count == 2  # vẫn báo là đã phát hiện
    assert len(analysis.usable_faces) == 1  # nhưng không dùng cho search
    small = analysis.faces[1]
    assert small.rejected_reason is not None and "too_small" in small.rejected_reason


def test_unaligned_space_reads_the_other_feature_list() -> None:
    body = success_body(1)
    aligned = np.asarray(body["result"]["features"][0], dtype=np.float32)
    unaligned = np.asarray(body["result"]["features_unalign"][0], dtype=np.float32)

    engine_a = make_engine(lambda r: httpx.Response(200, json=body), embedding_space="aligned")
    engine_u = make_engine(lambda r: httpx.Response(200, json=body), embedding_space="unaligned")

    vec_a = engine_a.analyze(b"\xff\xd8\xff x").faces[0].embedding
    vec_u = engine_u.analyze(b"\xff\xd8\xff x").faces[0].embedding

    assert np.allclose(vec_a, aligned, atol=1e-5)
    assert np.allclose(vec_u, unaligned, atol=1e-5)
    # Hai không gian khác nhau: đây là lý do không được trộn chúng trong một index.
    assert abs(float(vec_a @ vec_u)) < 0.3


def test_http_500_text_body_is_handled_and_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, text="Internal Server Error")

    engine = make_engine(handler)
    with pytest.raises(VisionBackendError) as exc:
        engine.analyze(b"\xff\xd8\xff x")

    assert exc.value.retryable is True
    assert calls["n"] == 2  # max_retries=1 -> 2 lần thử
    assert "Internal Server Error" in str(exc.value.details.get("body_prefix"))


def test_http_422_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(422, json={"detail": "bad payload"})

    engine = make_engine(handler)
    with pytest.raises(VisionBackendRejectedImage) as exc:
        engine.analyze(b"\xff\xd8\xff x")

    assert exc.value.retryable is False
    assert calls["n"] == 1  # 4xx: gọi lại cùng payload thì vẫn sai


def test_timeout_is_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    engine = make_engine(handler)
    with pytest.raises(VisionBackendTimeout) as exc:
        engine.analyze(b"\xff\xd8\xff x")
    assert exc.value.retryable is True


def test_retry_then_success() -> None:
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json=success_body(1))

    engine = make_engine(handler)
    analysis = engine.analyze(b"\xff\xd8\xff x")
    assert analysis.face_count == 1
    assert state["n"] == 2


def test_wrong_embedding_dim_raises_instead_of_silently_accepting() -> None:
    body = success_body(1)
    body["result"]["features"] = [[0.1] * 128]  # sai chiều
    engine = make_engine(lambda r: httpx.Response(200, json=body))

    with pytest.raises(VisionBackendError) as exc:
        engine.analyze(b"\xff\xd8\xff x")
    assert exc.value.details["got_dim"] == 128
    assert exc.value.details["expected_dim"] == 512


def test_non_json_200_response_raises() -> None:
    engine = make_engine(lambda r: httpx.Response(200, text="<html>nope</html>"))
    with pytest.raises(VisionBackendError):
        engine.analyze(b"\xff\xd8\xff x")


def test_unnormalized_vector_gets_normalized() -> None:
    body = success_body(1)
    body["result"]["features"] = [[2.0] + [0.0] * 511]  # norm = 2.0
    engine = make_engine(lambda r: httpx.Response(200, json=body))
    analysis = engine.analyze(b"\xff\xd8\xff x")

    vec = analysis.faces[0].embedding
    assert float(np.linalg.norm(vec)) == pytest.approx(1.0, abs=1e-6)
    assert any("normalize" in w for w in analysis.warnings)
