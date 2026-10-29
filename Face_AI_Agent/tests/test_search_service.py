"""Test logic quyết định của FaceSearchService.

``_classify`` là hàm thuần nhận danh sách ứng viên -> trạng thái, nên test dựng
score bằng tay, không cần index. Đây là phần quan trọng nhất của hệ thống về mặt
hậu quả: nó là chỗ một con số biến thành một kết luận về con người.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import MatchingConfig
from app.errors import ValidationFailed
from app.repositories.base import SearchHit
from app.services.search_service import (
    AMBIGUOUS,
    MATCH,
    NEEDS_REVIEW,
    NO_CANDIDATES,
    NO_MATCH,
    FaceSearchService,
)


def hit(score: float, rank: int = 1, subject: str = "s") -> SearchHit:
    return SearchHit(
        face_id=f"f{rank}",
        subject_id=f"{subject}{rank}",
        score=score,
        rank=rank,
        collection="c",
    )


@pytest.fixture
def service() -> FaceSearchService:
    config = MatchingConfig(
        match_threshold=0.55, review_band=0.05, ambiguous_margin=0.03, top_k=5
    )
    return FaceSearchService(repository=None, config=config)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Năm trạng thái
# --------------------------------------------------------------------------- #
def test_no_candidates_when_empty(service) -> None:
    result = service._classify(0, [])
    assert result.match_status == NO_CANDIDATES
    assert result.best_score is None
    # Phân biệt rõ với no_match: ta không so sánh với gì cả.
    assert "không có ứng viên" in result.reason


def test_clear_match(service) -> None:
    result = service._classify(0, [hit(0.90, 1), hit(0.40, 2)])
    assert result.match_status == MATCH
    assert result.best_score == 0.90
    assert result.margin == pytest.approx(0.50)


def test_single_candidate_above_threshold_is_match(service) -> None:
    result = service._classify(0, [hit(0.70, 1)])
    assert result.match_status == MATCH
    assert result.margin is None
    assert "chỉ có 1 ứng viên" in result.reason


def test_ambiguous_when_runner_up_too_close(service) -> None:
    """Hai ứng viên đều vượt ngưỡng và sát nhau -> KHÔNG được tự chọn top1."""
    result = service._classify(0, [hit(0.80, 1), hit(0.79, 2)])
    assert result.match_status == AMBIGUOUS
    assert result.margin == pytest.approx(0.01, abs=1e-6)
    assert "cần người xem lại" in result.reason


def test_margin_exactly_at_boundary_is_match(service) -> None:
    """margin == ambiguous_margin -> vẫn là match (ngưỡng dạng >=)."""
    result = service._classify(0, [hit(0.80, 1), hit(0.77, 2)])
    assert result.match_status == MATCH


def test_needs_review_just_below_threshold(service) -> None:
    result = service._classify(0, [hit(0.53, 1)])
    assert result.match_status == NEEDS_REVIEW
    assert "dải xem lại" in result.reason


def test_threshold_boundary_is_inclusive(service) -> None:
    assert service._classify(0, [hit(0.55, 1)]).match_status == MATCH
    assert service._classify(0, [hit(0.5499, 1)]).match_status == NEEDS_REVIEW


def test_review_band_lower_boundary(service) -> None:
    assert service._classify(0, [hit(0.50, 1)]).match_status == NEEDS_REVIEW
    assert service._classify(0, [hit(0.4999, 1)]).match_status == NO_MATCH


def test_no_match_far_below(service) -> None:
    result = service._classify(0, [hit(0.05, 1), hit(0.01, 2)])
    assert result.match_status == NO_MATCH
    assert result.best_score == 0.05


def test_reason_always_contains_the_numbers(service) -> None:
    """Mọi kết luận phải kèm căn cứ bằng số, không chỉ là một nhãn."""
    for hits in ([], [hit(0.9)], [hit(0.9), hit(0.89)], [hit(0.53)], [hit(0.1)]):
        result = service._classify(0, hits)
        assert result.reason, "mỗi trạng thái phải có lý do"
        if hits:
            assert f"{hits[0].score:.4f}" in result.reason


def test_public_dict_never_contains_probability_language(service) -> None:
    result = service._classify(0, [hit(0.9)])
    payload = result.to_public_dict()
    assert "similarity_score" in payload["candidates"][0]
    assert "probability" not in str(payload)
    assert "percent" not in str(payload)


# --------------------------------------------------------------------------- #
# Verification 1:1
# --------------------------------------------------------------------------- #
def test_verify_same_vector(service) -> None:
    vec = np.ones(8, dtype=np.float32) / np.sqrt(8)
    out = service.verify(vec, vec)
    assert out["similarity_score"] == pytest.approx(1.0, abs=1e-6)
    assert out["match_status"] == MATCH
    assert "không phải xác suất" in out["note"]


def test_verify_orthogonal_vectors(service) -> None:
    a = np.array([1, 0, 0, 0], dtype=np.float32)
    b = np.array([0, 1, 0, 0], dtype=np.float32)
    out = service.verify(a, b)
    assert out["similarity_score"] == pytest.approx(0.0, abs=1e-6)
    assert out["match_status"] == NO_MATCH


def test_verify_rejects_dim_mismatch(service) -> None:
    with pytest.raises(ValidationFailed):
        service.verify(np.ones(4, dtype=np.float32), np.ones(8, dtype=np.float32))


def test_verify_rejects_zero_vector(service) -> None:
    with pytest.raises(ValidationFailed):
        service.verify(np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32))


def test_verify_normalizes_before_comparing(service) -> None:
    """Vector chưa chuẩn hoá vẫn cho cosine đúng, không ra số > 1."""
    a = np.array([3.0, 0.0], dtype=np.float32)
    b = np.array([5.0, 0.0], dtype=np.float32)
    assert service.verify(a, b)["similarity_score"] == pytest.approx(1.0, abs=1e-6)
