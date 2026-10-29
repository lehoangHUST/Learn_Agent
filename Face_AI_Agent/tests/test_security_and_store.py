"""Test phân quyền, rate limit, result store (TTL + ACL) và redaction trong log."""

from __future__ import annotations

import logging
import time

import numpy as np
import pytest

from app.errors import AccessDenied, RateLimited, ResultNotFound, Unauthenticated
from app.logging_utils import JsonFormatter, _scrub
from app.repositories.result_store import KIND_DETECTION, KIND_SEARCH, SqliteResultStore
from app.security import (
    ROLE_ADMIN,
    ROLE_SEARCH,
    Principal,
    SlidingWindowRateLimiter,
    TokenPrincipalStore,
)


# --------------------------------------------------------------------------- #
# Token parsing
# --------------------------------------------------------------------------- #
def test_token_store_parses_roles_containing_colons() -> None:
    """Tên quyền chứa ``:`` nên dấu phân tách phải là ``|`` và ``,``.

    Dùng ``:`` làm phân tách sẽ cắt ``face:search`` thành ``face`` và ``search`` -
    principal nhận quyền sai mà không có lỗi nào được báo.
    """
    store = TokenPrincipalStore.from_env(
        "t-alice|alice|face:search,face:register|default,hr;t-adm|adm|admin|"
    )
    alice = store.resolve("t-alice")
    assert alice.user_id == "alice"
    assert alice.roles == {"face:search", "face:register"}
    assert alice.allowed_collections == {"default", "hr"}

    admin = store.resolve("t-adm")
    assert admin.is_admin is True
    assert admin.allowed_collections == frozenset()


def test_token_store_rejects_unknown_and_missing_tokens() -> None:
    store = TokenPrincipalStore.from_env("t|u|face:search|default")
    with pytest.raises(Unauthenticated):
        store.resolve("wrong-token")
    with pytest.raises(Unauthenticated):
        store.resolve(None)


def test_token_store_skips_malformed_entries() -> None:
    store = TokenPrincipalStore.from_env("justatoken;|nouser|;t|u|face:search|default")
    assert set(store.tokens) == {"t"}


# --------------------------------------------------------------------------- #
# Principal
# --------------------------------------------------------------------------- #
def test_admin_bypasses_role_and_collection_checks() -> None:
    admin = Principal(user_id="a", roles=frozenset({ROLE_ADMIN}))
    admin.require_role("face:anything")
    admin.require_collection("any-collection")
    admin.require_owner("someone-else")


def test_role_check_is_not_substring_based() -> None:
    p = Principal(user_id="u", roles=frozenset({"face:search"}))
    p.require_role("face:search")
    with pytest.raises(AccessDenied):
        p.require_role("face:register")


def test_require_owner_message_does_not_leak_content() -> None:
    p = Principal(user_id="bob", roles=frozenset({ROLE_SEARCH}))
    with pytest.raises(AccessDenied) as exc:
        p.require_owner("alice", what="result")
    assert "alice" not in str(exc.value)


# --------------------------------------------------------------------------- #
# Rate limit
# --------------------------------------------------------------------------- #
def test_rate_limiter_blocks_after_limit() -> None:
    limiter = SlidingWindowRateLimiter(limit_per_minute=3)
    for _ in range(3):
        limiter.check("user-a", now=100.0)
    with pytest.raises(RateLimited) as exc:
        limiter.check("user-a", now=100.0)
    assert exc.value.details["limit"] == 3
    assert exc.value.retryable is True


def test_rate_limiter_is_per_key() -> None:
    limiter = SlidingWindowRateLimiter(limit_per_minute=1)
    limiter.check("user-a", now=100.0)
    limiter.check("user-b", now=100.0)  # không bị ảnh hưởng bởi user-a


def test_rate_limiter_window_slides() -> None:
    limiter = SlidingWindowRateLimiter(limit_per_minute=1)
    limiter.check("user-a", now=100.0)
    with pytest.raises(RateLimited):
        limiter.check("user-a", now=150.0)
    limiter.check("user-a", now=161.0)  # quá 60s -> cửa sổ đã trượt


def test_rate_limiter_disabled_when_limit_is_zero() -> None:
    limiter = SlidingWindowRateLimiter(limit_per_minute=0)
    for _ in range(100):
        limiter.check("user-a", now=100.0)


# --------------------------------------------------------------------------- #
# Result store
# --------------------------------------------------------------------------- #
@pytest.fixture
def store(tmp_path) -> SqliteResultStore:
    return SqliteResultStore(tmp_path / "results.db", default_ttl_seconds=3600)


@pytest.fixture
def alice() -> Principal:
    return Principal(user_id="alice", roles=frozenset({ROLE_SEARCH}))


@pytest.fixture
def bob() -> Principal:
    return Principal(user_id="bob", roles=frozenset({ROLE_SEARCH}))


def test_save_and_read_roundtrip(store, alice) -> None:
    result_id = store.save(
        kind=KIND_DETECTION,
        owner_id="alice",
        payload={"total_face_count": 2},
        thread_id="thr-1",
        request_id="req-1",
        embeddings={0: np.ones(4, dtype=np.float32), 1: np.zeros(4, dtype=np.float32) + 2},
        embedding_space="aligned",
    )
    stored = store.get(result_id, principal=alice)
    assert stored.payload["total_face_count"] == 2
    assert stored.thread_id == "thr-1"

    vectors, space = store.get_embeddings(result_id, principal=alice)
    assert space == "aligned"
    assert sorted(vectors) == [0, 1]
    assert np.allclose(vectors[0], np.ones(4))
    assert np.allclose(vectors[1], np.full(4, 2.0))


def test_embeddings_are_writable_copies(store, alice) -> None:
    """``np.frombuffer`` cho view chỉ-đọc; phải trả bản copy để tầng trên normalize được."""
    result_id = store.save(
        kind=KIND_DETECTION,
        owner_id="alice",
        payload={},
        embeddings={0: np.ones(4, dtype=np.float32)},
        embedding_space="aligned",
    )
    vectors, _ = store.get_embeddings(result_id, principal=alice)
    vectors[0][0] = 5.0  # không được raise ValueError


def test_cross_owner_read_is_denied(store, alice, bob) -> None:
    result_id = store.save(kind=KIND_DETECTION, owner_id="alice", payload={"x": 1})
    with pytest.raises(AccessDenied):
        store.get(result_id, principal=bob)
    with pytest.raises(AccessDenied):
        store.get_embeddings(result_id, principal=bob)


def test_latest_filters_by_owner_in_sql(store, alice, bob) -> None:
    """Bob đoán đúng thread_id của Alice cũng không nhận được gì."""
    store.save(
        kind=KIND_DETECTION, owner_id="alice", payload={"x": 1}, thread_id="shared-thread"
    )
    assert store.latest(kind=KIND_DETECTION, principal=alice, thread_id="shared-thread")
    assert store.latest(kind=KIND_DETECTION, principal=bob, thread_id="shared-thread") is None


def test_latest_returns_most_recent(store, alice) -> None:
    first = store.save(kind=KIND_SEARCH, owner_id="alice", payload={"n": 1}, thread_id="t")
    time.sleep(0.01)
    second = store.save(kind=KIND_SEARCH, owner_id="alice", payload={"n": 2}, thread_id="t")
    latest = store.latest(kind=KIND_SEARCH, principal=alice, thread_id="t")
    assert latest is not None and latest.result_id == second
    assert latest.result_id != first


def test_kind_mismatch_is_not_found(store, alice) -> None:
    result_id = store.save(kind=KIND_DETECTION, owner_id="alice", payload={})
    with pytest.raises(ResultNotFound):
        store.get(result_id, principal=alice, kind=KIND_SEARCH)


def test_expired_result_is_treated_as_missing_and_purged(store, alice) -> None:
    result_id = store.save(
        kind=KIND_DETECTION, owner_id="alice", payload={"x": 1}, ttl_seconds=-1
    )
    with pytest.raises(ResultNotFound) as exc:
        store.get(result_id, principal=alice)
    assert "hết hạn" in exc.value.message
    # Đã bị dọn, không chỉ bị ẩn.
    assert store.health()["results_total"] == 0


def test_purge_expired_removes_embeddings_too(store, alice) -> None:
    store.save(
        kind=KIND_DETECTION,
        owner_id="alice",
        payload={},
        embeddings={0: np.ones(4, dtype=np.float32)},
        embedding_space="aligned",
        ttl_seconds=-1,
    )
    assert store.purge_expired() == 1
    assert store.health()["results_total"] == 0


def test_delete_for_owner_requires_ownership(store, alice, bob) -> None:
    store.save(kind=KIND_DETECTION, owner_id="alice", payload={})
    with pytest.raises(AccessDenied):
        store.delete_for_owner("alice", principal=bob)
    assert store.delete_for_owner("alice", principal=alice) == 1


def test_admin_can_read_any_owner(store) -> None:
    admin = Principal(user_id="root", roles=frozenset({ROLE_ADMIN}))
    result_id = store.save(kind=KIND_DETECTION, owner_id="alice", payload={"x": 1})
    assert store.get(result_id, principal=admin).owner_id == "alice"


def test_saving_embeddings_without_space_is_rejected(store) -> None:
    with pytest.raises(ValueError):
        store.save(
            kind=KIND_DETECTION,
            owner_id="alice",
            payload={},
            embeddings={0: np.ones(4, dtype=np.float32)},
        )


# --------------------------------------------------------------------------- #
# Redaction trong log
# --------------------------------------------------------------------------- #
def test_scrub_redacts_biometric_and_secret_fields() -> None:
    payload = {
        "base64_image": "A" * 5000,
        "features": [[0.1] * 512],
        "embedding": [0.1, 0.2],
        "api_key": "sk-secret",
        "face_count": 3,
    }
    scrubbed = _scrub(payload)
    assert "A" * 100 not in str(scrubbed)
    assert "sk-secret" not in str(scrubbed)
    assert "0.1" not in str(scrubbed)
    # Nhưng vẫn giữ *hình dạng* để debug được.
    assert "len=5000" in scrubbed["base64_image"]
    assert scrubbed["face_count"] == 3


def test_scrub_truncates_any_long_string() -> None:
    """Chuỗi dài bị cắt bất kể tên key - ảnh có thể tới dưới một tên key lạ."""
    scrubbed = _scrub({"some_unexpected_field": "B" * 9000})
    assert "truncated" in scrubbed["some_unexpected_field"]


def test_scrub_handles_nested_and_long_lists() -> None:
    scrubbed = _scrub({"items": [{"base64": "x" * 400} for _ in range(50)]})
    assert "more" in str(scrubbed["items"][-1])


def test_json_formatter_scrubs_extra_fields() -> None:
    record = logging.LogRecord(
        name="t", level=logging.INFO, pathname="p", lineno=1, msg="event", args=(), exc_info=None
    )
    record.base64_image = "Z" * 3000
    record.request_id = "req-1"
    out = JsonFormatter().format(record)
    assert "Z" * 100 not in out
    assert "req-1" in out
