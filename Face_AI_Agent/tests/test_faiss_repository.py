"""Test kho vector FAISS: cosine đúng, persistence đúng, phân quyền đúng."""

from __future__ import annotations

import numpy as np
import pytest

from app.errors import AccessDenied, VectorStoreError
from app.repositories.base import FaceRecord
from app.repositories.faiss_repository import FaissFaceRepository
from app.security import ROLE_REGISTER, ROLE_SEARCH, Principal
from app.vision.fake import identity_vector

DIM = 512
COLLECTION = "test-collection"


def record(face_id: str, subject: str, identity: str, **kwargs) -> FaceRecord:
    return FaceRecord(
        face_id=face_id,
        subject_id=subject,
        embedding=identity_vector(identity, DIM),
        embedding_space=kwargs.pop("embedding_space", "aligned"),
        collection=kwargs.pop("collection", COLLECTION),
        owner_id=kwargs.pop("owner_id", "user-alice"),
        **kwargs,
    )


@pytest.fixture
def repo(tmp_path) -> FaissFaceRepository:
    return FaissFaceRepository(tmp_path / "index", dim=DIM, embedding_space="aligned")


@pytest.fixture
def owner() -> Principal:
    return Principal(
        user_id="user-alice",
        roles=frozenset({ROLE_SEARCH, ROLE_REGISTER, "face:delete", "face:read_metadata"}),
        allowed_collections=frozenset({COLLECTION}),
    )


# --------------------------------------------------------------------------- #
def test_empty_index_returns_empty_results_not_error(repo, owner) -> None:
    """"Chưa có dữ liệu" là câu trả lời hợp lệ, khác hẳn "đã tìm và không thấy"."""
    query = identity_vector("someone", DIM).reshape(1, -1)
    hits = repo.search(query, principal=owner, top_k=5, collection=COLLECTION)
    assert hits == [[]]
    assert repo.stats(COLLECTION).total_faces == 0


def test_self_match_scores_one(repo, owner) -> None:
    repo.register([record("f1", "subject-A", "A")], principal=owner)
    query = identity_vector("A", DIM).reshape(1, -1)
    hits = repo.search(query, principal=owner, top_k=5, collection=COLLECTION)

    assert len(hits[0]) == 1
    assert hits[0][0].score == pytest.approx(1.0, abs=1e-5)
    assert hits[0][0].subject_id == "subject-A"
    assert hits[0][0].rank == 1


def test_different_identity_scores_near_zero(repo, owner) -> None:
    repo.register([record("f1", "subject-A", "A")], principal=owner)
    hits = repo.search(
        identity_vector("totally-different", DIM).reshape(1, -1),
        principal=owner,
        top_k=5,
        collection=COLLECTION,
    )
    assert abs(hits[0][0].score) < 0.2


def test_batch_search_returns_one_row_per_query(repo, owner) -> None:
    repo.register(
        [record("f1", "subject-A", "A"), record("f2", "subject-B", "B")], principal=owner
    )
    queries = np.vstack([identity_vector("A", DIM), identity_vector("B", DIM)])
    hits = repo.search(queries, principal=owner, top_k=2, collection=COLLECTION)

    assert len(hits) == 2
    assert hits[0][0].subject_id == "subject-A"
    assert hits[1][0].subject_id == "subject-B"


def test_ranking_is_ordered_descending(repo, owner) -> None:
    repo.register(
        [
            record("f1", "subject-A", "A"),
            record("f2", "subject-B", "B"),
            record("f3", "subject-C", "C"),
        ],
        principal=owner,
    )
    hits = repo.search(
        identity_vector("A", DIM).reshape(1, -1),
        principal=owner,
        top_k=3,
        collection=COLLECTION,
    )
    scores = [h.score for h in hits[0]]
    assert scores == sorted(scores, reverse=True)
    assert [h.rank for h in hits[0]] == [1, 2, 3]


def test_upsert_replaces_vector_for_same_face_id(repo, owner) -> None:
    repo.register([record("f1", "subject-A", "A")], principal=owner)
    repo.register([record("f1", "subject-A-renamed", "Z")], principal=owner)

    assert repo.stats(COLLECTION).total_faces == 1
    hits = repo.search(
        identity_vector("Z", DIM).reshape(1, -1),
        principal=owner,
        top_k=5,
        collection=COLLECTION,
    )
    assert hits[0][0].subject_id == "subject-A-renamed"
    assert hits[0][0].score == pytest.approx(1.0, abs=1e-5)


def test_delete_removes_from_index(repo, owner) -> None:
    repo.register([record("f1", "subject-A", "A")], principal=owner)
    assert repo.delete(["f1"], principal=owner) == 1
    assert repo.stats(COLLECTION).total_faces == 0


def test_persistence_survives_reload(tmp_path, owner) -> None:
    first = FaissFaceRepository(tmp_path / "index", dim=DIM, embedding_space="aligned")
    first.register(
        [record("f1", "subject-A", "A"), record("f2", "subject-B", "B")], principal=owner
    )

    reloaded = FaissFaceRepository(tmp_path / "index", dim=DIM, embedding_space="aligned")
    assert reloaded.stats(COLLECTION).total_faces == 2
    assert reloaded.stats(COLLECTION).total_subjects == 2
    hits = reloaded.search(
        identity_vector("A", DIM).reshape(1, -1),
        principal=owner,
        top_k=1,
        collection=COLLECTION,
    )
    assert hits[0][0].subject_id == "subject-A"


def test_reload_with_different_embedding_space_refuses_to_start(tmp_path, owner) -> None:
    """Đây là bất biến quan trọng nhất của kho này.

    Trộn 'aligned' và 'unaligned' cho similarity ~ 0.04 giữa hai ảnh CÙNG một người,
    nên mọi kết quả đều sai mà không có dấu hiệu. Thà không nạp được index còn hơn.
    """
    first = FaissFaceRepository(tmp_path / "index", dim=DIM, embedding_space="aligned")
    first.register([record("f1", "subject-A", "A")], principal=owner)

    with pytest.raises(VectorStoreError) as exc:
        FaissFaceRepository(tmp_path / "index", dim=DIM, embedding_space="unaligned")
    assert "không so sánh được" in str(exc.value)


def test_registering_wrong_space_is_rejected(repo, owner) -> None:
    with pytest.raises(VectorStoreError):
        repo.register(
            [record("f1", "subject-A", "A", embedding_space="unaligned")], principal=owner
        )


def test_wrong_query_dim_is_rejected(repo, owner) -> None:
    repo.register([record("f1", "subject-A", "A")], principal=owner)
    with pytest.raises(VectorStoreError):
        repo.search(
            np.zeros((1, 128), dtype=np.float32),
            principal=owner,
            top_k=1,
            collection=COLLECTION,
        )


def test_degenerate_vectors_are_rejected(repo, owner) -> None:
    bad = FaceRecord(
        face_id="f-zero",
        subject_id="s",
        embedding=np.zeros(DIM, dtype=np.float32),
        embedding_space="aligned",
        collection=COLLECTION,
    )
    with pytest.raises(VectorStoreError):
        repo.register([bad], principal=owner)


def test_unnormalized_vector_is_normalized_at_write(repo, owner) -> None:
    vec = identity_vector("A", DIM) * 5.0
    rec = FaceRecord(
        face_id="f1",
        subject_id="subject-A",
        embedding=vec,
        embedding_space="aligned",
        collection=COLLECTION,
    )
    repo.register([rec], principal=owner)
    hits = repo.search(
        identity_vector("A", DIM).reshape(1, -1),
        principal=owner,
        top_k=1,
        collection=COLLECTION,
    )
    # Nếu không normalize thì score sẽ là ~5.0 - vô nghĩa với một ngưỡng cosine.
    assert hits[0][0].score == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# Phân quyền
# --------------------------------------------------------------------------- #
def test_search_requires_allowed_collection(repo, owner) -> None:
    repo.register([record("f1", "subject-A", "A")], principal=owner)
    outsider = Principal(
        user_id="user-bob",
        roles=frozenset({ROLE_SEARCH}),
        allowed_collections=frozenset({"other-collection"}),
    )
    with pytest.raises(AccessDenied):
        repo.search(
            identity_vector("A", DIM).reshape(1, -1),
            principal=outsider,
            top_k=1,
            collection=COLLECTION,
        )


def test_register_requires_role(repo) -> None:
    reader = Principal(
        user_id="reader",
        roles=frozenset({ROLE_SEARCH}),
        allowed_collections=frozenset({COLLECTION}),
    )
    with pytest.raises(AccessDenied):
        repo.register([record("f1", "subject-A", "A")], principal=reader)


def test_private_records_hidden_from_other_users(repo, owner) -> None:
    repo.register(
        [record("f1", "subject-A", "A", visibility="private", owner_id="user-alice")],
        principal=owner,
    )
    other = Principal(
        user_id="user-bob",
        roles=frozenset({ROLE_SEARCH}),
        allowed_collections=frozenset({COLLECTION}),
    )
    hits = repo.search(
        identity_vector("A", DIM).reshape(1, -1),
        principal=other,
        top_k=5,
        collection=COLLECTION,
    )
    assert hits[0] == []


def test_metadata_hidden_without_role(repo, owner) -> None:
    repo.register(
        [record("f1", "subject-A", "A", metadata={"employee_id": "E-123"})], principal=owner
    )
    no_meta = Principal(
        user_id="user-carol",
        roles=frozenset({ROLE_SEARCH}),
        allowed_collections=frozenset({COLLECTION}),
    )
    hits = repo.search(
        identity_vector("A", DIM).reshape(1, -1),
        principal=no_meta,
        top_k=1,
        collection=COLLECTION,
    )
    assert hits[0][0].metadata == {}
    assert hits[0][0].metadata_visible is False
    # Vẫn thấy subject_id và score - chỉ metadata bị ẩn.
    assert hits[0][0].subject_id == "subject-A"


def test_metadata_filter_narrows_results(repo, owner) -> None:
    repo.register(
        [
            record("f1", "subject-A", "A", metadata={"dept": "eng"}),
            record("f2", "subject-B", "B", metadata={"dept": "sales"}),
        ],
        principal=owner,
    )
    hits = repo.search(
        identity_vector("B", DIM).reshape(1, -1),
        principal=owner,
        top_k=5,
        collection=COLLECTION,
        metadata_filter={"dept": "eng"},
    )
    assert [h.subject_id for h in hits[0]] == ["subject-A"]


def test_metadata_filter_supports_list_membership(repo, owner) -> None:
    repo.register(
        [
            record("f1", "subject-A", "A", metadata={"dept": "eng"}),
            record("f2", "subject-B", "B", metadata={"dept": "sales"}),
        ],
        principal=owner,
    )
    hits = repo.search(
        identity_vector("A", DIM).reshape(1, -1),
        principal=owner,
        top_k=5,
        collection=COLLECTION,
        metadata_filter={"dept": ["eng", "sales"]},
    )
    assert len(hits[0]) == 2
