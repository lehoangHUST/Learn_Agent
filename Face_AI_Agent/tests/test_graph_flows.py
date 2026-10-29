"""Test luồng end-to-end qua LangGraph - 10 tình huống trong yêu cầu.

Mỗi test chạy graph thật (compile, checkpoint, routing, node) với engine giả. Không
mock node nào: nếu routing sai hay reducer sai thì những test này đỏ.

Danh sách tình huống (theo đúng yêu cầu):

1. Ảnh không có khuôn mặt
2. Ảnh có một khuôn mặt
3. Ảnh có nhiều khuôn mặt
4. Ảnh lỗi / không hợp lệ
5. Vector database không có dữ liệu
6. Có kết quả nhưng không đạt threshold
7. Có nhiều kết quả gần nhau
8. Người dùng hỏi tiếp về kết quả ở lượt trước
9. Người dùng cố truy cập kết quả của session khác
10. Một node bị lỗi trong quá trình thực thi graph
"""

from __future__ import annotations

import numpy as np
import pytest

from app.agent.runner import FaceAgentRunner
from app.errors import AccessDenied, ImageDecodeFailed, ValidationFailed, VisionBackendError
from app.security import Principal
from app.services.search_service import AMBIGUOUS, MATCH, NO_CANDIDATES, NO_MATCH
from app.vision.fake import FakeFace, FakeImagePlan, identity_vector
from tests.conftest import COLLECTION, b64, make_image_bytes


# --------------------------------------------------------------------------- #
# 1. Ảnh không có khuôn mặt
# --------------------------------------------------------------------------- #
def test_case_01_image_with_no_face(runner: FaceAgentRunner, engine) -> None:
    data = make_image_bytes(seed=10)
    engine.set_plan(data, FakeImagePlan(faces=[]))

    result = runner.run(
        principal=_alice(),
        query="Có bao nhiêu khuôn mặt trong ảnh?",
        images_base64=[b64(data)],
    )
    out = result.output

    # Không có mặt KHÔNG phải lỗi: request thành công, câu trả lời nói rõ là 0.
    assert out["status"] == "succeeded"
    assert out["errors"] == []
    assert out["detection_summary"]["total_face_count"] == 0
    assert "Không phát hiện được khuôn mặt nào" in out["answer"]
    # Và nói cả nguyên nhân thường gặp, thay vì chỉ "0".
    assert any("quá nhỏ" in w or "không có mặt người" in w for w in out["warnings"])


# --------------------------------------------------------------------------- #
# 2. Ảnh có một khuôn mặt
# --------------------------------------------------------------------------- #
def test_case_02_single_face(runner: FaceAgentRunner, engine) -> None:
    data = make_image_bytes(seed=11)
    engine.set_plan(data, FakeImagePlan(faces=[FakeFace(identity="synthetic-A")]))

    out = runner.run(
        principal=_alice(),
        query="Phát hiện khuôn mặt trong ảnh",
        images_base64=[b64(data)],
    ).output

    assert out["status"] == "succeeded"
    assert out["intent"] == "face_detection"
    assert out["detection_summary"]["total_face_count"] == 1
    face = out["detection_summary"]["faces"][0]
    assert len(face["box"]) == 4
    assert len(face["landmarks"]) == 5
    assert face["global_face_index"] == 0
    # Không chạy search vì kế hoạch không có bước đó.
    assert "search_faces" not in out["executed_steps"]
    assert out["search_result_id"] is None


def test_case_02b_confidence_question_is_answered_honestly(
    runner: FaceAgentRunner, engine
) -> None:
    """Hỏi về confidence -> phải nói là KHÔNG CÓ, không được bịa số."""
    data = make_image_bytes(seed=12)
    engine.set_plan(
        data,
        FakeImagePlan(
            faces=[
                FakeFace(identity="A", box=(0, 0, 100, 120)),
                FakeFace(identity="B", box=(200, 200, 260, 270)),
            ]
        ),
    )

    out = runner.run(
        principal=_alice(),
        query="Khuôn mặt nào có confidence cao nhất?",
        images_base64=[b64(data)],
    ).output

    assert out["detection_summary"]["detector_confidence_available"] is False
    for face in out["detection_summary"]["faces"]:
        assert face["confidence"] is None
    assert "KHÔNG trả confidence" in out["answer"] or "không trả confidence" in out["answer"]
    # Dữ kiện thay thế (kích thước) có, nhưng được nói rõ là không phải confidence.
    ranking = out["analysis"]["detection"]["face_size_ranking"]
    assert [r["global_face_index"] for r in ranking] == [0, 1]
    assert "KHÔNG phải" in out["analysis"]["detection"]["size_is_not_confidence"]


# --------------------------------------------------------------------------- #
# 3. Ảnh có nhiều khuôn mặt
# --------------------------------------------------------------------------- #
def test_case_03_multiple_faces(runner: FaceAgentRunner, engine) -> None:
    data = make_image_bytes(seed=13)
    engine.set_plan(
        data,
        FakeImagePlan(
            faces=[
                FakeFace(identity="A", box=(10, 10, 60, 70)),     # nhỏ nhất
                FakeFace(identity="B", box=(100, 100, 300, 320)),  # to nhất
                FakeFace(identity="C", box=(400, 100, 500, 220)),
            ]
        ),
    )

    out = runner.run(
        principal=_alice(),
        query="Có bao nhiêu khuôn mặt?",
        images_base64=[b64(data)],
    ).output

    assert out["detection_summary"]["total_face_count"] == 3
    indices = [f["global_face_index"] for f in out["detection_summary"]["faces"]]
    assert indices == [0, 1, 2]
    areas = [f["size"]["area_px"] for f in out["detection_summary"]["faces"]]
    assert areas == sorted(areas, reverse=True), "index phải ổn định theo diện tích giảm dần"
    assert "3 khuôn mặt" in out["answer"]


# --------------------------------------------------------------------------- #
# 4. Ảnh lỗi / không hợp lệ
# --------------------------------------------------------------------------- #
def test_case_04_invalid_image_fails_before_the_graph(runner: FaceAgentRunner) -> None:
    """Ảnh sai bị chặn ở validate, TRƯỚC khi graph chạy và trước khi gọi model.

    Lỗi 4xx được raise ra ngoài để tầng API trả đúng status - không đi vào graph rồi
    trả 200 kèm một lỗi lẫn trong body.
    """
    with pytest.raises(ImageDecodeFailed):
        runner.run(
            principal=_alice(),
            query="Phát hiện khuôn mặt",
            images_base64=["khong-phai-base64!!!"],
        )


def test_case_04b_non_image_payload_rejected(runner: FaceAgentRunner) -> None:
    import base64 as _b64

    payload = _b64.b64encode(b"#!/bin/sh\necho pwned\n" * 30).decode()
    with pytest.raises(ValidationFailed) as exc:
        runner.run(principal=_alice(), query="Phát hiện khuôn mặt", images_base64=[payload])
    assert exc.value.code in ("unsupported_image_type", "image_decode_failed")


def test_case_04c_engine_never_called_for_invalid_image(runner: FaceAgentRunner, engine) -> None:
    before = engine.call_count
    with pytest.raises(ValidationFailed):
        runner.run(principal=_alice(), query="x", images_base64=["!!!"])
    assert engine.call_count == before, "ảnh sai không được tốn một lời gọi backend nào"


# --------------------------------------------------------------------------- #
# 5. Vector database không có dữ liệu
# --------------------------------------------------------------------------- #
def test_case_05_empty_index(runner: FaceAgentRunner, engine) -> None:
    data = make_image_bytes(seed=15)
    engine.set_plan(data, FakeImagePlan(faces=[FakeFace(identity="A")]))

    out = runner.run(
        principal=_alice(),
        query="Tìm kiếm khuôn mặt trong ảnh này",
        images_base64=[b64(data)],
    ).output

    assert out["status"] == "succeeded"
    assert out["search_summary"]["query_face_count"] == 1
    assert out["search_summary"]["results"][0]["match_status"] == NO_CANDIDATES
    assert out["search_summary"]["index"]["total_faces"] == 0
    # "Chưa có dữ liệu" phải được nói khác với "đã tìm và không thấy ai giống".
    assert "KHÁC" in out["analysis"]["search"]["empty_index"]


# --------------------------------------------------------------------------- #
# 6. Có kết quả nhưng không đạt threshold
# --------------------------------------------------------------------------- #
def test_case_06_candidates_below_threshold(
    runner: FaceAgentRunner, engine, register_subject
) -> None:
    register_subject("subject-known", seed=20, identity="identity-known")

    query_image = make_image_bytes(seed=21)
    engine.set_plan(
        query_image, FakeImagePlan(faces=[FakeFace(identity="identity-completely-different")])
    )

    out = runner.run(
        principal=_alice(),
        query="Tìm kiếm khuôn mặt này",
        images_base64=[b64(query_image)],
    ).output

    result = out["search_summary"]["results"][0]
    assert result["match_status"] == NO_MATCH
    assert result["candidates"], "vẫn trả về ứng viên gần nhất để người dùng tự đánh giá"
    assert result["best_score"] < out["search_summary"]["threshold"]["match_threshold"]
    assert "thấp hơn" in result["reason"]


def test_case_06b_needs_review_band(runner: FaceAgentRunner, engine, deps, principal) -> None:
    """Score nằm ngay dưới ngưỡng -> needs_review, không phải no_match."""
    target = identity_vector("base-identity", 512)
    # Dựng một vector có cosine ~ 0.52 với target (ngưỡng 0.55, review_band 0.05).
    other = identity_vector("orthogonal-ish", 512)
    mixed = 0.52 * target + np.sqrt(1 - 0.52**2) * other
    mixed = mixed / np.linalg.norm(mixed)

    registered_image = make_image_bytes(seed=22)
    engine.set_plan(
        registered_image, FakeImagePlan(faces=[FakeFace(identity="x", embedding=target)])
    )
    analysis = engine.analyze(registered_image)
    deps.search.register_faces(
        analysis.faces,
        subject_id="subject-borderline",
        principal=principal,
        collection=COLLECTION,
        embedding_space=deps.embedding.embedding_space,
        consent_ref="consent://synthetic/borderline",
    )

    query_image = make_image_bytes(seed=23)
    engine.set_plan(query_image, FakeImagePlan(faces=[FakeFace(identity="y", embedding=mixed)]))

    out = runner.run(
        principal=_alice(), query="Tìm kiếm khuôn mặt", images_base64=[b64(query_image)]
    ).output

    result = out["search_summary"]["results"][0]
    assert result["best_score"] == pytest.approx(0.52, abs=0.02)
    assert result["match_status"] == "needs_review"
    assert "dải xem lại" in result["reason"]


# --------------------------------------------------------------------------- #
# 7. Có nhiều kết quả gần nhau
# --------------------------------------------------------------------------- #
def test_case_07_ambiguous_close_candidates(
    runner: FaceAgentRunner, engine, deps, principal
) -> None:
    """Hai người đã đăng ký có embedding gần như trùng nhau -> ambiguous.

    Hệ thống KHÔNG được tự chọn ứng viên đầu: đây đúng là trường hợp mà một kết luận
    tự động gây hậu quả.
    """
    base = identity_vector("twin-base", 512)
    noise = identity_vector("twin-noise", 512)
    twin_a = base
    twin_b = base + 0.02 * noise
    twin_b = twin_b / np.linalg.norm(twin_b)

    for subject, vector, seed in (("subject-twin-A", twin_a, 30), ("subject-twin-B", twin_b, 31)):
        img = make_image_bytes(seed=seed)
        engine.set_plan(img, FakeImagePlan(faces=[FakeFace(identity=subject, embedding=vector)]))
        analysis = engine.analyze(img)
        deps.search.register_faces(
            analysis.faces,
            subject_id=subject,
            principal=principal,
            collection=COLLECTION,
            embedding_space=deps.embedding.embedding_space,
            consent_ref=f"consent://synthetic/{subject}",
        )

    query_image = make_image_bytes(seed=32)
    engine.set_plan(
        query_image, FakeImagePlan(faces=[FakeFace(identity="q", embedding=twin_a)])
    )

    out = runner.run(
        principal=_alice(), query="Tìm kiếm khuôn mặt này", images_base64=[b64(query_image)]
    ).output

    result = out["search_summary"]["results"][0]
    assert result["match_status"] == AMBIGUOUS
    assert result["margin"] < out["search_summary"]["threshold"]["ambiguous_margin"]
    assert len(result["candidates"]) == 2
    assert "cần người xem lại" in result["reason"]
    assert "không phân biệt được" in out["answer"]


def test_case_07b_clear_match_when_margin_is_wide(
    runner: FaceAgentRunner, engine, register_subject
) -> None:
    """Đối chứng: cùng identity, không có ai gần -> match rõ ràng."""
    register_subject("subject-solo", seed=33, identity="identity-solo")
    register_subject("subject-other", seed=34, identity="identity-very-different")

    query_image = make_image_bytes(seed=35)
    engine.set_plan(
        query_image,
        FakeImagePlan(faces=[FakeFace(identity="identity-solo", jitter=0.1)]),
    )

    out = runner.run(
        principal=_alice(), query="Tìm kiếm khuôn mặt này", images_base64=[b64(query_image)]
    ).output

    result = out["search_summary"]["results"][0]
    assert result["match_status"] == MATCH
    assert result["candidates"][0]["subject_id"] == "subject-solo"
    assert result["best_score"] > 0.9  # jitter=0.1 -> cos ~ 0.995
    # Dù là match, cảnh báo ngưỡng chưa hiệu chuẩn vẫn phải xuất hiện.
    assert any("hiệu chuẩn" in w for w in out["warnings"])


# --------------------------------------------------------------------------- #
# 8. Hỏi tiếp về kết quả lượt trước
# --------------------------------------------------------------------------- #
def test_case_08_followup_conversation(
    runner: FaceAgentRunner, engine, register_subject
) -> None:
    """Đúng kịch bản trong yêu cầu, 3 lượt trên cùng một thread.

    Lượt 2 và 3 KHÔNG gửi lại ảnh: embedding được lấy từ result store của lượt 1.
    """
    register_subject("subject-target", seed=40, identity="identity-face-2")

    data = make_image_bytes(seed=41)
    engine.set_plan(
        data,
        FakeImagePlan(
            faces=[
                FakeFace(identity="identity-face-1", box=(0, 0, 200, 220)),
                FakeFace(identity="identity-face-2", box=(300, 0, 440, 160)),
                FakeFace(identity="identity-face-3", box=(0, 300, 100, 420)),
            ]
        ),
    )

    alice = _alice()

    # --- Lượt 1: gửi ảnh, hỏi số khuôn mặt ---
    turn1 = runner.run(
        principal=alice, query="Có bao nhiêu khuôn mặt?", images_base64=[b64(data)]
    )
    thread_id = turn1.thread_id
    assert turn1.output["detection_summary"]["total_face_count"] == 3
    assert "3 khuôn mặt" in turn1.answer

    # --- Lượt 2: "tìm kiếm khuôn mặt thứ hai", KHÔNG gửi ảnh ---
    turn2 = runner.run(
        principal=alice, query="Tìm kiếm khuôn mặt thứ hai", thread_id=thread_id
    )
    out2 = turn2.output
    assert out2["intent"] == "face_search"
    # Không detect lại: dùng embedding đã lưu.
    assert "detect_faces" not in out2["executed_steps"]
    assert "generate_embeddings" in out2["executed_steps"]
    assert out2["search_summary"]["query_face_count"] == 1
    result2 = out2["search_summary"]["results"][0]
    assert result2["face_index"] == 1, "'thứ hai' phải là face_index 1"
    assert result2["match_status"] == MATCH
    assert result2["candidates"][0]["subject_id"] == "subject-target"

    # --- Lượt 3: hỏi similarity của kết quả, KHÔNG gửi ảnh, KHÔNG chạy model ---
    turn3 = runner.run(
        principal=alice,
        query="Kết quả thứ nhất có similarity bao nhiêu?",
        thread_id=thread_id,
    )
    out3 = turn3.output
    assert out3["intent"] == "result_question"
    assert "detect_faces" not in out3["executed_steps"]
    assert "search_faces" not in out3["executed_steps"]
    # Con số trong lượt 3 phải TRÙNG KHỚP với lượt 2 - vì cả hai đọc cùng một kết quả
    # đã lưu, không phải đọc lại văn bản câu trả lời trước.
    score_turn2 = result2["best_score"]
    assert f"{score_turn2:.4f}" in out3["answer"]
    assert out3["analysis"]["loaded_from_result_ids"]


def test_case_08b_followup_without_prior_result_explains_what_is_missing(
    runner: FaceAgentRunner,
) -> None:
    """Thread mới, hỏi về kết quả cũ -> giải thích thiếu gì, không im lặng lỗi."""
    out = runner.run(
        principal=_alice(), query="Kết quả thứ hai có similarity bao nhiêu?"
    ).output

    assert out["status"] in ("failed", "partial")
    assert out["errors"]
    assert out["errors"][0]["retryable"] is False
    assert "ảnh" in out["answer"].lower()


def test_case_08c_requesting_nonexistent_face_index_is_explained(
    runner: FaceAgentRunner, engine
) -> None:
    data = make_image_bytes(seed=42)
    engine.set_plan(data, FakeImagePlan(faces=[FakeFace(identity="only-one")]))

    turn1 = runner.run(
        principal=_alice(), query="Có bao nhiêu khuôn mặt?", images_base64=[b64(data)]
    )
    out = runner.run(
        principal=_alice(),
        query="Tìm kiếm khuôn mặt thứ năm",
        thread_id=turn1.thread_id,
    ).output

    assert out["errors"]
    assert out["errors"][0]["code"] == "invalid_input"
    # Nói rõ ảnh có mấy mặt, không chỉ "index không hợp lệ".
    assert out["errors"][0]["details"]["available"] == [0]


# --------------------------------------------------------------------------- #
# 9. Truy cập kết quả của session khác
# --------------------------------------------------------------------------- #
def test_case_09_cross_session_access_is_denied(
    runner: FaceAgentRunner, engine, deps, other_principal
) -> None:
    """Bob biết thread_id và result_id của Alice nhưng vẫn không đọc được."""
    data = make_image_bytes(seed=50)
    engine.set_plan(data, FakeImagePlan(faces=[FakeFace(identity="alice-face")]))

    alice_turn = runner.run(
        principal=_alice(), query="Có bao nhiêu khuôn mặt?", images_base64=[b64(data)]
    )
    alice_result_id = alice_turn.output["detection_result_id"]
    assert alice_result_id

    # Đọc trực tiếp từ store: 403.
    with pytest.raises(AccessDenied):
        deps.results.get(alice_result_id, principal=other_principal)
    with pytest.raises(AccessDenied):
        deps.results.get_embeddings(alice_result_id, principal=other_principal)

    # Qua graph, dùng chính thread_id của Alice: Bob không thấy kết quả nào.
    bob_out = runner.run(
        principal=other_principal,
        query="Kết quả thứ hai có similarity bao nhiêu?",
        thread_id=alice_turn.thread_id,
    ).output
    assert bob_out["status"] in ("failed", "partial")
    assert bob_out["search_summary"] is None
    assert bob_out["detection_summary"] is None

    # Và dùng target_result_id tường minh của Alice cũng không lọt.
    bob_targeted = runner.run(
        principal=other_principal,
        query="Tìm kiếm khuôn mặt thứ nhất",
        thread_id=alice_turn.thread_id,
        target_result_id=alice_result_id,
    ).output
    assert bob_targeted["status"] in ("failed", "partial")
    assert bob_targeted["search_summary"] is None
    assert any(e["code"] in ("access_denied", "invalid_input") for e in bob_targeted["errors"])


def test_case_09b_owner_still_has_access(runner: FaceAgentRunner, engine, deps) -> None:
    """Đối chứng với test trên: chính chủ vẫn đọc được."""
    data = make_image_bytes(seed=51)
    engine.set_plan(data, FakeImagePlan(faces=[FakeFace(identity="a")]))
    alice = _alice()

    turn = runner.run(principal=alice, query="Đếm khuôn mặt", images_base64=[b64(data)])
    stored = deps.results.get(turn.output["detection_result_id"], principal=alice)
    assert stored.payload["total_face_count"] == 1


def test_case_09c_admin_can_read_any_result(
    runner: FaceAgentRunner, engine, deps, admin_principal
) -> None:
    data = make_image_bytes(seed=52)
    engine.set_plan(data, FakeImagePlan(faces=[FakeFace(identity="a")]))
    turn = runner.run(principal=_alice(), query="Đếm khuôn mặt", images_base64=[b64(data)])

    stored = deps.results.get(turn.output["detection_result_id"], principal=admin_principal)
    assert stored.owner_id == "user-alice"


# --------------------------------------------------------------------------- #
# 10. Một node bị lỗi
# --------------------------------------------------------------------------- #
def test_case_10_retryable_node_failure_is_retried_then_succeeds(
    runner: FaceAgentRunner, engine, deps
) -> None:
    """Lỗi retry được -> handle_error quay lại node đó, lần hai thành công."""
    data = make_image_bytes(seed=60)
    engine.set_plan(
        data,
        FakeImagePlan(
            faces=[FakeFace(identity="A")],
            raise_error=VisionBackendError("backend tạm thời lỗi"),
        ),
    )

    calls = {"n": 0}
    original = engine.analyze

    def flaky(image_bytes: bytes):
        calls["n"] += 1
        if calls["n"] == 1:
            raise VisionBackendError("backend tạm thời lỗi")
        engine.set_plan(image_bytes, FakeImagePlan(faces=[FakeFace(identity="A")]))
        return original(image_bytes)

    engine.analyze = flaky  # type: ignore[method-assign]
    engine.detect = flaky  # type: ignore[method-assign]

    out = runner.run(
        principal=_alice(), query="Phát hiện khuôn mặt", images_base64=[b64(data)]
    ).output

    assert "handle_error" in out["executed_steps"]
    assert out["executed_steps"].count("detect_faces") == 2, "phải chạy lại detect_faces"
    assert out["status"] == "partial"  # có kết quả, nhưng cũng có lỗi đã xảy ra
    assert out["detection_summary"]["total_face_count"] == 1
    assert any("thử lại" in w for w in out["warnings"])


def test_case_10b_non_retryable_failure_still_returns_an_explanation(
    runner: FaceAgentRunner, engine
) -> None:
    """Lỗi không retry được -> vẫn đi qua answer để giải thích, không trả response rỗng."""
    data = make_image_bytes(seed=61)
    engine.set_plan(
        data,
        FakeImagePlan(
            faces=[],
            raise_error=ValidationFailed("backend từ chối ảnh này"),
        ),
    )

    out = runner.run(
        principal=_alice(), query="Phát hiện khuôn mặt", images_base64=[b64(data)]
    ).output

    assert out["status"] == "failed"
    assert out["errors"][0]["retryable"] is False
    assert out["errors"][0]["node"] == "detect_faces"
    assert "handle_error" in out["executed_steps"]
    assert "answer_question" in out["executed_steps"]
    assert "finalize_response" in out["executed_steps"]
    assert out["answer"], "phải có câu trả lời giải thích, không được rỗng"
    assert "Không thực hiện được yêu cầu" in out["answer"]


def test_case_10c_retry_is_bounded(runner: FaceAgentRunner, engine, deps) -> None:
    """Lỗi retry được nhưng lỗi mãi -> dừng sau max_node_retries, không vòng vô hạn."""
    data = make_image_bytes(seed=62)

    def always_fail(image_bytes: bytes):
        raise VisionBackendError("backend chết hẳn")

    engine.analyze = always_fail  # type: ignore[method-assign]
    engine.detect = always_fail  # type: ignore[method-assign]

    out = runner.run(
        principal=_alice(), query="Phát hiện khuôn mặt", images_base64=[b64(data)]
    ).output

    max_retries = deps.settings.graph.max_node_retries
    assert out["executed_steps"].count("detect_faces") == max_retries + 1
    assert out["status"] == "failed"
    assert out["answer"]
    assert any("retry" in w or "thử lại" in w for w in out["warnings"])


def test_case_10d_unexpected_exception_does_not_escape_the_graph(
    runner: FaceAgentRunner, engine
) -> None:
    """Exception lạ (không phải FaceAIError) cũng phải thành response đúng schema."""
    data = make_image_bytes(seed=63)

    def boom(image_bytes: bytes):
        raise RuntimeError("lỗi hoàn toàn không lường trước")

    engine.analyze = boom  # type: ignore[method-assign]
    engine.detect = boom  # type: ignore[method-assign]

    out = runner.run(
        principal=_alice(), query="Phát hiện khuôn mặt", images_base64=[b64(data)]
    ).output

    assert out["status"] == "failed"
    assert out["errors"][0]["code"] == "internal_error"
    assert out["errors"][0]["details"]["exception_type"] == "RuntimeError"
    assert out["answer"]


# --------------------------------------------------------------------------- #
# Những bất biến chung của mọi luồng
# --------------------------------------------------------------------------- #
def test_unsupported_request_runs_no_model(runner: FaceAgentRunner, engine) -> None:
    data = make_image_bytes(seed=70)
    engine.set_plan(data, FakeImagePlan(faces=[FakeFace(identity="A")]))
    before = engine.call_count

    out = runner.run(
        principal=_alice(),
        query="Người trong ảnh bao nhiêu tuổi?",
        images_base64=[b64(data)],
    ).output

    assert out["intent"] == "unsupported_request"
    assert engine.call_count == before, "yêu cầu bị từ chối không được tốn một lời gọi model"
    assert out["errors"][0]["code"] == "unsupported_request"
    assert "không suy" in out["answer"].lower() or "không suy" in out["errors"][0]["message"]


def test_general_question_runs_no_model(runner: FaceAgentRunner, engine) -> None:
    before = engine.call_count
    out = runner.run(
        principal=_alice(),
        query="Giải thích sự khác biệt giữa confidence của detector và similarity score",
    ).output

    assert out["intent"] == "general_face_question"
    assert engine.call_count == before
    assert out["status"] == "succeeded"
    assert "Face Detection" in out["answer"]


def test_response_never_contains_embedding_values(
    runner: FaceAgentRunner, engine, register_subject
) -> None:
    """Bất biến bảo mật: vector sinh trắc học không bao giờ ra khỏi hệ thống."""
    register_subject("subject-A", seed=80, identity="identity-A")
    data = make_image_bytes(seed=81)
    engine.set_plan(data, FakeImagePlan(faces=[FakeFace(identity="identity-A")]))

    out = runner.run(
        principal=_alice(), query="Tìm kiếm khuôn mặt này", images_base64=[b64(data)]
    ).output

    text = repr(out)
    for key in ("embedding\":", "features", "vector\":"):
        assert key not in text, f"{key} không được xuất hiện trong output"
    assert out["detection_summary"]["faces"][0]["has_embedding"] is True  # chỉ có cờ boolean


def test_single_backend_call_per_image_in_search_flow(
    runner: FaceAgentRunner, engine, register_subject
) -> None:
    """Detect + embed + search chỉ tốn ĐÚNG MỘT lời gọi backend cho một ảnh.

    Đây là lý do node detect lưu luôn embedding thay vì để node embed gọi lại.
    """
    register_subject("subject-A", seed=90, identity="identity-A")
    data = make_image_bytes(seed=91)
    engine.set_plan(data, FakeImagePlan(faces=[FakeFace(identity="identity-A")]))

    before = engine.call_count
    out = runner.run(
        principal=_alice(), query="Tìm kiếm khuôn mặt này", images_base64=[b64(data)]
    ).output

    assert "search_faces" in out["executed_steps"]
    assert engine.call_count - before == 1


def test_executed_steps_accumulate_in_order(runner: FaceAgentRunner, engine) -> None:
    data = make_image_bytes(seed=92)
    engine.set_plan(data, FakeImagePlan(faces=[FakeFace(identity="A")]))

    out = runner.run(
        principal=_alice(), query="Phát hiện khuôn mặt", images_base64=[b64(data)]
    ).output

    steps = out["executed_steps"]
    assert steps[:3] == ["validate_input", "classify_intent", "plan_execution"]
    assert steps[-1] == "finalize_response"
    # Field tích lũy: mỗi node nối đúng một phần tử, không ghi đè phần của node khác.
    assert len(steps) == len(set(steps)) or steps.count("detect_faces") >= 1


def test_multiple_images_get_global_face_indices(runner: FaceAgentRunner, engine) -> None:
    first = make_image_bytes(seed=93)
    second = make_image_bytes(seed=94)
    engine.set_plan(first, FakeImagePlan(faces=[FakeFace(identity="A", box=(0, 0, 100, 110))]))
    engine.set_plan(
        second,
        FakeImagePlan(
            faces=[
                FakeFace(identity="B", box=(0, 0, 120, 130)),
                FakeFace(identity="C", box=(200, 0, 280, 90)),
            ]
        ),
    )

    out = runner.run(
        principal=_alice(),
        query="Có bao nhiêu khuôn mặt?",
        images_base64=[b64(first), b64(second)],
    ).output

    assert out["detection_summary"]["image_count"] == 2
    assert out["detection_summary"]["total_face_count"] == 3
    assert [f["global_face_index"] for f in out["detection_summary"]["faces"]] == [0, 1, 2]
    assert [f["image_index"] for f in out["detection_summary"]["faces"]] == [0, 1, 1]


def _alice() -> Principal:
    from app.security import (
        ROLE_DELETE,
        ROLE_READ_METADATA,
        ROLE_REGISTER,
        ROLE_SEARCH,
    )

    return Principal(
        user_id="user-alice",
        roles=frozenset({ROLE_SEARCH, ROLE_REGISTER, ROLE_DELETE, ROLE_READ_METADATA}),
        allowed_collections=frozenset({COLLECTION}),
    )
