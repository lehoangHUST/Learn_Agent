"""Test tầng HTTP: schema, status code, xác thực, rate limit, hình dạng lỗi.

Dùng ``dependency_overrides`` của FastAPI để tiêm ``AgentDeps`` dựng từ engine giả.
``TestClient`` được tạo **không** dùng context manager để lifespan không chạy - nếu
chạy, lifespan sẽ dựng dependency thật và gọi tới backend GPU trong mạng LAN.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.agent.graph import build_agent_graph
from app.agent.runner import FaceAgentRunner
from app.api.deps import (
    deps_dependency,
    principal_dependency,
    rate_limit_dependency,
    runner_dependency,
)
from app.main import app
from app.security import TokenPrincipalStore
from tests.conftest import COLLECTION, b64, make_image_bytes


@pytest.fixture
def client(deps, principal):
    from langgraph.checkpoint.memory import InMemorySaver

    runner = FaceAgentRunner(build_agent_graph(checkpointer=InMemorySaver()), deps)

    app.dependency_overrides[deps_dependency] = lambda: deps
    app.dependency_overrides[principal_dependency] = lambda: principal
    app.dependency_overrides[rate_limit_dependency] = lambda: principal
    app.dependency_overrides[runner_dependency] = lambda: runner
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        app.dependency_overrides.clear()


# --------------------------------------------------------------------------- #
# /faces/detect
# --------------------------------------------------------------------------- #
def test_detect_returns_declared_schema(client, engine) -> None:
    data = make_image_bytes(seed=100)
    response = client.post("/api/v1/faces/detect", json={"images_base64": [b64(data)]})

    assert response.status_code == 200
    body = response.json()
    assert body["total_face_count"] == 1
    assert body["detector_confidence_available"] is False
    assert "không phải confidence" in body["confidence_note"].lower()
    assert body["embedding_dim"] == 512
    face = body["faces"][0]
    assert face["confidence"] is None
    assert len(face["box"]) == 4
    assert len(face["landmarks"]) == 5
    assert "X-Request-Id" in response.headers


def test_detect_response_never_leaks_embedding(client) -> None:
    data = make_image_bytes(seed=101)
    body = client.post("/api/v1/faces/detect", json={"images_base64": [b64(data)]}).json()
    text = str(body)
    assert "features" not in text
    assert "embedding\":" not in text
    assert body["faces"][0]["has_embedding"] is True  # chỉ cờ boolean


def test_detect_rejects_invalid_base64_with_error_envelope(client) -> None:
    response = client.post("/api/v1/faces/detect", json={"images_base64": ["!!!not-base64"]})
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "image_decode_failed"
    assert body["error"]["retryable"] is False
    assert body["request_id"]


def test_detect_rejects_oversized_image(client, deps) -> None:
    deps.settings.limits.max_image_bytes = 100
    response = client.post(
        "/api/v1/faces/detect", json={"images_base64": [b64(make_image_bytes(seed=102))]}
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "image_too_large"


def test_detect_rejects_empty_image_list_with_schema_error(client) -> None:
    response = client.post("/api/v1/faces/detect", json={"images_base64": []})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "schema_validation"


def test_schema_validation_error_does_not_echo_the_image(client) -> None:
    """Body lỗi của pydantic mặc định chứa cả ``input`` - với ảnh base64 thì đó là
    cả tấm ảnh nằm trong response lỗi và trong log."""
    payload = b64(make_image_bytes(seed=103))
    response = client.post("/api/v1/faces/detect", json={"images_base64": payload})
    assert response.status_code == 422
    assert payload[:50] not in response.text


# --------------------------------------------------------------------------- #
# /faces/register + /faces/search
# --------------------------------------------------------------------------- #
def test_register_then_search_finds_the_subject(client, engine) -> None:
    from app.vision.fake import FakeFace, FakeImagePlan

    enrolled = make_image_bytes(seed=110)
    engine.set_plan(enrolled, FakeImagePlan(faces=[FakeFace(identity="identity-X")]))

    created = client.post(
        "/api/v1/faces/register",
        json={
            "image_base64": b64(enrolled),
            "subject_id": "subject-X",
            "metadata": {"dept": "eng"},
            "consent_ref": "consent://synthetic/X",
        },
    )
    assert created.status_code == 201
    assert created.json()["has_consent_ref"] is True
    assert created.json()["index_stats"]["total_faces"] == 1

    query = make_image_bytes(seed=111)
    engine.set_plan(
        query, FakeImagePlan(faces=[FakeFace(identity="identity-X", jitter=0.1)])
    )
    found = client.post("/api/v1/faces/search", json={"image_base64": b64(query)})

    assert found.status_code == 200
    body = found.json()
    assert body["query_face_count"] == 1
    result = body["results"][0]
    assert result["match_status"] == "match"
    assert result["candidates"][0]["subject_id"] == "subject-X"
    assert result["candidates"][0]["metadata"] == {
        "dept": "eng",
        "source_face_index": 0,
        "box": result["candidates"][0]["metadata"]["box"],
        "image_sha256": result["candidates"][0]["metadata"]["image_sha256"],
    }
    assert body["threshold"]["calibrated"] is False
    assert "không phải xác suất" in body["score_semantics"].lower()


def test_register_multi_face_image_requires_explicit_index(client, engine) -> None:
    from app.vision.fake import FakeFace, FakeImagePlan

    data = make_image_bytes(seed=112)
    engine.set_plan(
        data,
        FakeImagePlan(
            faces=[
                FakeFace(identity="A", box=(0, 0, 100, 110)),
                FakeFace(identity="B", box=(200, 0, 290, 100)),
            ]
        ),
    )
    response = client.post(
        "/api/v1/faces/register",
        json={"image_base64": b64(data), "subject_id": "subject-ambiguous"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["details"]["available_face_indices"] == [0, 1]


def test_search_on_empty_index_returns_no_candidates(client) -> None:
    response = client.post(
        "/api/v1/faces/search", json={"image_base64": b64(make_image_bytes(seed=113))}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["match_status"] == "no_candidates"
    assert body["index"]["total_faces"] == 0
    assert any("chưa có khuôn mặt nào" in w for w in body["warnings"])


# --------------------------------------------------------------------------- #
# /agent/chat
# --------------------------------------------------------------------------- #
def test_chat_returns_thread_id_for_followup(client, engine) -> None:
    data = make_image_bytes(seed=120)
    response = client.post(
        "/api/v1/agent/chat",
        json={"message": "Có bao nhiêu khuôn mặt trong ảnh?", "images_base64": [b64(data)]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "succeeded"
    assert body["intent"] == "face_detection"
    assert body["intent_source"] == "rule"
    assert body["answer_source"] == "template"
    assert body["thread_id"].startswith("thr_")
    assert body["executed_steps"][0] == "validate_input"
    assert body["detection_result_id"]
    assert body["search_result_id"] is None


def test_chat_followup_uses_same_thread(client, engine, register_subject) -> None:
    from app.vision.fake import FakeFace, FakeImagePlan

    register_subject("subject-target", seed=121, identity="identity-target")
    data = make_image_bytes(seed=122)
    engine.set_plan(
        data,
        FakeImagePlan(
            faces=[
                FakeFace(identity="identity-other", box=(0, 0, 200, 220)),
                FakeFace(identity="identity-target", box=(300, 0, 440, 160)),
            ]
        ),
    )

    first = client.post(
        "/api/v1/agent/chat",
        json={"message": "Có bao nhiêu khuôn mặt?", "images_base64": [b64(data)]},
    ).json()
    thread_id = first["thread_id"]

    second = client.post(
        "/api/v1/agent/chat",
        json={"message": "Tìm kiếm khuôn mặt thứ hai", "thread_id": thread_id},
    ).json()

    assert second["thread_id"] == thread_id
    assert second["intent"] == "face_search"
    assert "detect_faces" not in second["executed_steps"]
    assert second["search_summary"]["results"][0]["face_index"] == 1
    # executed_steps của lượt 2 không được mang theo các bước của lượt 1.
    assert second["executed_steps"].count("validate_input") == 1


def test_chat_run_scoped_state_does_not_leak_between_turns(client, engine) -> None:
    """Lượt 1 lỗi, lượt 2 thành công -> lượt 2 không được mang theo lỗi của lượt 1."""
    from app.vision.fake import FakeFace, FakeImagePlan

    bad = make_image_bytes(seed=123)
    engine.set_plan(bad, FakeImagePlan(faces=[], raise_error=RuntimeError("boom")))

    first = client.post(
        "/api/v1/agent/chat",
        json={"message": "Phát hiện khuôn mặt", "images_base64": [b64(bad)]},
    ).json()
    assert first["status"] == "failed"
    assert first["errors"]

    good = make_image_bytes(seed=124)
    engine.set_plan(good, FakeImagePlan(faces=[FakeFace(identity="A")]))
    second = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "Phát hiện khuôn mặt",
            "images_base64": [b64(good)],
            "thread_id": first["thread_id"],
        },
    ).json()

    assert second["status"] == "succeeded"
    assert second["errors"] == []


def test_chat_rejects_unsupported_request(client, engine) -> None:
    data = make_image_bytes(seed=125)
    body = client.post(
        "/api/v1/agent/chat",
        json={"message": "Người này bao nhiêu tuổi?", "images_base64": [b64(data)]},
    ).json()

    assert body["intent"] == "unsupported_request"
    assert body["status"] == "failed"
    assert body["errors"][0]["code"] == "unsupported_request"


def test_chat_requires_non_empty_message(client) -> None:
    response = client.post("/api/v1/agent/chat", json={"message": ""})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "schema_validation"


# --------------------------------------------------------------------------- #
# /tasks, /results, /threads
# --------------------------------------------------------------------------- #
def test_task_lookup_by_request_id(client, engine) -> None:
    data = make_image_bytes(seed=130)
    chat = client.post(
        "/api/v1/agent/chat",
        json={"message": "Đếm khuôn mặt", "images_base64": [b64(data)]},
    ).json()

    task = client.get(f"/api/v1/tasks/{chat['request_id']}").json()
    assert task["status"] == "succeeded"
    assert task["thread_id"] == chat["thread_id"]
    assert task["results"][0]["kind"] == "detection"


def test_task_lookup_unknown_request_id(client) -> None:
    body = client.get("/api/v1/tasks/req_does_not_exist").json()
    assert body["status"] == "not_found"
    assert "hết hạn" in body["note"]


def test_explain_endpoint_says_what_would_change_the_result(
    client, engine, register_subject
) -> None:
    from app.vision.fake import FakeFace, FakeImagePlan

    register_subject("subject-known", seed=131, identity="identity-known")
    query = make_image_bytes(seed=132)
    engine.set_plan(query, FakeImagePlan(faces=[FakeFace(identity="identity-unrelated")]))

    search = client.post(
        "/api/v1/faces/search", json={"image_base64": b64(query)}
    ).json()

    explained = client.get(f"/api/v1/results/{search['result_id']}/explain").json()
    assert explained["threshold_calibrated"] is False
    explanation = explained["explanations"][0]
    assert explanation["match_status"] == "no_match"
    assert "cần thêm" in explanation["what_would_change_it"]
    assert "không nên chuyển thành phần trăm" in explained["score_semantics"]


def test_thread_results_listing(client, engine) -> None:
    data = make_image_bytes(seed=133)
    chat = client.post(
        "/api/v1/agent/chat",
        json={"message": "Đếm khuôn mặt", "images_base64": [b64(data)]},
    ).json()

    listing = client.get(f"/api/v1/threads/{chat['thread_id']}/results").json()
    assert listing["thread_id"] == chat["thread_id"]
    assert listing["results"][0]["kind"] == "detection"


def test_reading_another_users_result_is_denied(client, deps, engine, other_principal) -> None:
    data = make_image_bytes(seed=134)
    chat = client.post(
        "/api/v1/agent/chat",
        json={"message": "Đếm khuôn mặt", "images_base64": [b64(data)]},
    ).json()

    # Đổi principal sang Bob rồi đọc result_id của Alice.
    app.dependency_overrides[principal_dependency] = lambda: other_principal
    response = client.get(f"/api/v1/results/{chat['detection_result_id']}")
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "access_denied"


# --------------------------------------------------------------------------- #
# /health
# --------------------------------------------------------------------------- #
def test_health_surfaces_uncalibrated_threshold(client) -> None:
    body = client.get("/api/v1/health").json()
    assert body["matching_threshold_calibrated"] is False
    assert body["status"] == "degraded"  # index rỗng + ngưỡng chưa hiệu chuẩn
    assert any("hiệu chuẩn" in w for w in body["warnings"])
    assert any("chưa có khuôn mặt nào" in w for w in body["warnings"])
    names = {c["name"] for c in body["components"]}
    assert names == {
        "vision_backend",
        "vector_store",
        "result_store",
        "image_store",
        "llm",
    }


def test_health_does_not_leak_secrets(client, deps) -> None:
    deps.settings.openai_api_key = "sk-super-secret-value"
    body = client.get("/api/v1/health").text
    assert "sk-super-secret-value" not in body


# --------------------------------------------------------------------------- #
# Xác thực & rate limit (dùng dependency thật, không override)
# --------------------------------------------------------------------------- #
@pytest.fixture
def auth_client(deps):
    """Client dùng ``principal_dependency`` thật để test xác thực."""
    from langgraph.checkpoint.memory import InMemorySaver

    runner = FaceAgentRunner(build_agent_graph(checkpointer=InMemorySaver()), deps)
    app.dependency_overrides[deps_dependency] = lambda: deps
    app.dependency_overrides[runner_dependency] = lambda: runner
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        app.dependency_overrides.clear()


def test_missing_token_rejected_when_tokens_configured(auth_client, deps) -> None:
    deps.tokens = TokenPrincipalStore.from_env(
        f"tok-alice|user-alice|face:search,face:register|{COLLECTION}"
    )
    response = auth_client.post(
        "/api/v1/faces/detect", json={"images_base64": [b64(make_image_bytes(seed=140))]}
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthenticated"


def test_valid_token_is_accepted(auth_client, deps) -> None:
    deps.tokens = TokenPrincipalStore.from_env(
        f"tok-alice|user-alice|face:search,face:register|{COLLECTION}"
    )
    response = auth_client.post(
        "/api/v1/faces/detect",
        json={"images_base64": [b64(make_image_bytes(seed=141))]},
        headers={"Authorization": "Bearer tok-alice"},
    )
    assert response.status_code == 200


def test_malformed_authorization_header_rejected(auth_client, deps) -> None:
    deps.tokens = TokenPrincipalStore.from_env(f"tok|user|face:search|{COLLECTION}")
    response = auth_client.post(
        "/api/v1/faces/detect",
        json={"images_base64": [b64(make_image_bytes(seed=142))]},
        headers={"Authorization": "tok"},  # thiếu 'Bearer'
    )
    assert response.status_code == 401


def test_anonymous_access_refused_outside_dev(auth_client, deps) -> None:
    """Token chưa cấu hình + môi trường không phải dev -> 401, không tự mở cửa."""
    deps.tokens = TokenPrincipalStore(tokens={})
    deps.settings.environment = "prod"
    response = auth_client.post(
        "/api/v1/faces/detect", json={"images_base64": [b64(make_image_bytes(seed=143))]}
    )
    assert response.status_code == 401
    assert "ẩn danh" in response.json()["error"]["message"]


def test_rate_limit_returns_429_with_retry_after(auth_client, deps) -> None:
    from app.security import SlidingWindowRateLimiter

    deps.tokens = TokenPrincipalStore.from_env(f"tok|user-x|face:search|{COLLECTION}")
    deps.rate_limiter = SlidingWindowRateLimiter(limit_per_minute=2)
    payload = {"images_base64": [b64(make_image_bytes(seed=144))]}
    headers = {"Authorization": "Bearer tok"}

    assert auth_client.post("/api/v1/faces/detect", json=payload, headers=headers).status_code == 200
    assert auth_client.post("/api/v1/faces/detect", json=payload, headers=headers).status_code == 200
    third = auth_client.post("/api/v1/faces/detect", json=payload, headers=headers)
    assert third.status_code == 429
    body = third.json()
    assert body["error"]["code"] == "rate_limited"
    assert body["error"]["retryable"] is True
    assert body["error"]["details"]["retry_after_seconds"] > 0


# --------------------------------------------------------------------------- #
# OpenAPI
# --------------------------------------------------------------------------- #
def test_openapi_documents_the_caveats(client) -> None:
    """Swagger phải nói trước những điều dễ hiểu sai, không để người dùng tự phát hiện."""
    spec = client.get("/openapi.json").json()
    description = spec["info"]["description"]
    assert "không trả confidence" in description
    assert "cosine similarity" in description
    assert "chưa được hiệu chuẩn" in description
    paths = set(spec["paths"])
    assert {
        "/api/v1/faces/detect",
        "/api/v1/faces/search",
        "/api/v1/faces/register",
        "/api/v1/agent/chat",
        "/api/v1/tasks/{request_id}",
        "/api/v1/health",
    } <= paths
