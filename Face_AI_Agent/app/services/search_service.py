"""Face Search Service - nơi duy nhất biến *khoảng cách* thành *kết luận*.

Bốn khái niệm, bốn thứ khác nhau
-------------------------------
* **Face Detection**   - khuôn mặt nằm ở đâu trong ảnh. Trả bounding box.
* **Face Embedding**   - biểu diễn một khuôn mặt thành vector. Không có khái niệm
  "đúng/sai", chỉ có không gian vector.
* **Face Verification** - 1:1. Hai mẫu này có cùng một người không? Một điểm số,
  một ngưỡng. Xem ``FaceSearchService.verify``.
* **Face Identification / Search** - 1:N. Trong tập đã đăng ký, những ai gần mẫu
  truy vấn nhất? Trả **danh sách ứng viên có xếp hạng**, không trả "danh tính".

Similarity KHÔNG phải xác suất
------------------------------
``score`` là cosine similarity trong [-1, 1]. Nó không phải P(cùng một người).
Muốn có xác suất thì phải hiệu chuẩn (Platt scaling / isotonic regression) trên một
tập validation đại diện cho đúng dân số và điều kiện ảnh của hệ thống đích - và
ngay cả khi đó, xác suất phụ thuộc tỷ lệ tiên nghiệm. Hệ thống này cố tình **không**
đưa ra con số phần trăm nào, và ``explanation`` luôn nói rõ điều đó.

Ba trạng thái, không phải hai
-----------------------------
Buộc chọn giữa "match" và "no_match" là nguồn gốc của phần lớn sai lầm có hậu quả.
Ở đây có thêm vùng cần người xem lại:

* ``match``        - top1 >= threshold VÀ cách biệt với top2 >= ambiguous_margin.
* ``ambiguous``    - top1 >= threshold nhưng top2 sát quá: hai người trông giống
                     nhau, hoặc index có ảnh trùng. Không được tự chọn top1.
* ``needs_review`` - top1 nằm trong dải [threshold - review_band, threshold).
* ``no_match``     - top1 dưới cả dải đó.
* ``no_candidates``- index rỗng hoặc filter loại hết. Khác hẳn ``no_match``: ta
                     không so sánh với gì cả, nên không kết luận được gì.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.config import MatchingConfig
from app.errors import ValidationFailed, VectorStoreError
from app.logging_utils import timed
from app.repositories.base import BaseFaceSearchRepository, FaceRecord, SearchHit
from app.security import ROLE_REGISTER, ROLE_SEARCH, Principal
from app.services.embedding_service import EmbeddingBatch
from app.vision.base import DetectedFace

logger = logging.getLogger(__name__)

MATCH = "match"
AMBIGUOUS = "ambiguous"
NEEDS_REVIEW = "needs_review"
NO_MATCH = "no_match"
NO_CANDIDATES = "no_candidates"


@dataclass(slots=True)
class FaceMatchResult:
    """Kết quả so khớp cho MỘT khuôn mặt truy vấn."""

    face_index: int
    match_status: str
    hits: list[SearchHit] = field(default_factory=list)
    best_score: float | None = None
    runner_up_score: float | None = None
    margin: float | None = None
    threshold: float = 0.0
    reason: str = ""

    @property
    def best_hit(self) -> SearchHit | None:
        return self.hits[0] if self.hits else None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "face_index": self.face_index,
            "match_status": self.match_status,
            "best_score": None if self.best_score is None else round(self.best_score, 6),
            "runner_up_score": None if self.runner_up_score is None else round(self.runner_up_score, 6),
            "margin": None if self.margin is None else round(self.margin, 6),
            "threshold": round(self.threshold, 6),
            "reason": self.reason,
            "candidates": [
                {
                    "rank": h.rank,
                    "face_id": h.face_id,
                    "subject_id": h.subject_id,
                    "similarity_score": round(h.score, 6),
                    "collection": h.collection,
                    "metadata": h.metadata,
                    "metadata_visible": h.metadata_visible,
                }
                for h in self.hits
            ],
        }


@dataclass(slots=True)
class SearchOutcome:
    """Kết quả so khớp cho cả một ảnh (có thể nhiều mặt)."""

    results: list[FaceMatchResult]
    collection: str
    metric: str
    threshold: float
    review_band: float
    ambiguous_margin: float
    top_k: int
    threshold_calibrated: bool
    index_total_faces: int
    index_total_subjects: int
    skipped_faces: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def query_face_count(self) -> int:
        return len(self.results)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "query_face_count": self.query_face_count,
            "collection": self.collection,
            "threshold": {
                "metric": self.metric,
                "match_threshold": round(self.threshold, 6),
                "review_band": round(self.review_band, 6),
                "ambiguous_margin": round(self.ambiguous_margin, 6),
                "calibrated": self.threshold_calibrated,
            },
            "top_k": self.top_k,
            "index": {
                "total_faces": self.index_total_faces,
                "total_subjects": self.index_total_subjects,
            },
            "results": [r.to_public_dict() for r in self.results],
            "skipped_faces": list(self.skipped_faces),
            "warnings": list(self.warnings),
            "score_semantics": (
                "similarity_score là cosine similarity trong [-1, 1] giữa hai embedding, "
                "KHÔNG phải xác suất cùng danh tính."
            ),
        }


class FaceSearchService:
    def __init__(self, repository: BaseFaceSearchRepository, config: MatchingConfig) -> None:
        self._repo = repository
        self._config = config

    # ------------------------------------------------------------------ #
    # 1:N - identification / search
    # ------------------------------------------------------------------ #
    def search(
        self,
        batch: EmbeddingBatch,
        *,
        principal: Principal,
        collection: str,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> SearchOutcome:
        principal.require_role(ROLE_SEARCH)
        k = top_k or self._config.top_k
        stats = self._repo.stats(collection)

        warnings: list[str] = []
        if not self._config.calibrated:
            warnings.append(
                "match_threshold CHƯA được hiệu chuẩn cho model này "
                f"({self._config.calibration_note}). Mọi kết luận 'match' chỉ nên coi "
                "là ứng viên cần người xác nhận."
            )
        if batch.embedding_space != self._repo.embedding_space:
            raise VectorStoreError(
                f"vector truy vấn thuộc không gian {batch.embedding_space!r} nhưng index "
                f"dùng {self._repo.embedding_space!r}",
                details={
                    "query_space": batch.embedding_space,
                    "index_space": self._repo.embedding_space,
                },
            )

        if batch.is_empty:
            return SearchOutcome(
                results=[],
                collection=collection,
                metric=self._config.metric,
                threshold=self._config.match_threshold,
                review_band=self._config.review_band,
                ambiguous_margin=self._config.ambiguous_margin,
                top_k=k,
                threshold_calibrated=self._config.calibrated,
                index_total_faces=stats.total_faces,
                index_total_subjects=stats.total_subjects,
                skipped_faces=list(batch.skipped),
                warnings=[*warnings, "không có khuôn mặt nào dùng được để truy vấn"],
            )

        with timed(
            logger,
            "face_search",
            collection=collection,
            query_faces=batch.count,
            top_k=k,
            index_total=stats.total_faces,
            **principal.log_fields(),
        ) as extra:
            # Một lời gọi cho tất cả mặt: nhanh hơn n lời gọi và không cần fan-out
            # song song trong graph.
            per_query_hits = self._repo.search(
                batch.vectors,
                principal=principal,
                top_k=k,
                collection=collection,
                metadata_filter=metadata_filter,
            )
            extra["hits_total"] = sum(len(h) for h in per_query_hits)

        if stats.total_faces == 0:
            warnings.append(
                f"collection {collection!r} chưa có khuôn mặt nào được đăng ký - "
                "không thể so khớp. Dùng scripts/register_face.py để nạp dữ liệu."
            )

        results = [
            self._classify(face_index, hits)
            # strict=True: repository phải trả đúng một hàng kết quả cho mỗi vector
            # truy vấn. Lệch độ dài ở đây nghĩa là kết quả bị gán cho sai khuôn mặt
            # - loại bug không được phép im lặng trong một hệ thống nhận dạng.
            for face_index, hits in zip(batch.face_indices, per_query_hits, strict=True)
        ]

        return SearchOutcome(
            results=results,
            collection=collection,
            metric=self._config.metric,
            threshold=self._config.match_threshold,
            review_band=self._config.review_band,
            ambiguous_margin=self._config.ambiguous_margin,
            top_k=k,
            threshold_calibrated=self._config.calibrated,
            index_total_faces=stats.total_faces,
            index_total_subjects=stats.total_subjects,
            skipped_faces=list(batch.skipped),
            warnings=warnings,
        )

    def _classify(self, face_index: int, hits: list[SearchHit]) -> FaceMatchResult:
        """Biến danh sách ứng viên thành một trong năm trạng thái.

        Thuần, không I/O, không phụ thuộc thời gian -> test trực tiếp được bằng các
        bộ score dựng tay, không cần index.
        """
        threshold = self._config.match_threshold

        if not hits:
            return FaceMatchResult(
                face_index=face_index,
                match_status=NO_CANDIDATES,
                threshold=threshold,
                reason=(
                    "không có ứng viên nào để so sánh (index rỗng, hoặc bộ lọc "
                    "metadata/quyền truy cập đã loại hết)"
                ),
            )

        best = hits[0]
        runner_up = hits[1].score if len(hits) > 1 else None
        margin = None if runner_up is None else best.score - runner_up

        if best.score >= threshold:
            if margin is not None and margin < self._config.ambiguous_margin:
                status = AMBIGUOUS
                reason = (
                    f"top1 ({best.score:.4f}) vượt ngưỡng {threshold:.4f} nhưng chỉ hơn "
                    f"top2 ({runner_up:.4f}) {margin:.4f} < margin "
                    f"{self._config.ambiguous_margin:.4f}: hai ứng viên gần như không "
                    "phân biệt được, cần người xem lại"
                )
            else:
                status = MATCH
                reason = (
                    f"top1 ({best.score:.4f}) >= ngưỡng {threshold:.4f}"
                    + (f" và hơn top2 {margin:.4f}" if margin is not None else " (chỉ có 1 ứng viên)")
                )
        elif best.score >= threshold - self._config.review_band:
            status = NEEDS_REVIEW
            reason = (
                f"top1 ({best.score:.4f}) nằm trong dải xem lại "
                f"[{threshold - self._config.review_band:.4f}, {threshold:.4f}): "
                "dưới ngưỡng nhưng không đủ thấp để kết luận không khớp"
            )
        else:
            status = NO_MATCH
            reason = (
                f"top1 ({best.score:.4f}) thấp hơn cả dải xem lại "
                f"({threshold - self._config.review_band:.4f})"
            )

        return FaceMatchResult(
            face_index=face_index,
            match_status=status,
            hits=hits,
            best_score=best.score,
            runner_up_score=runner_up,
            margin=margin,
            threshold=threshold,
            reason=reason,
        )

    # ------------------------------------------------------------------ #
    # 1:1 - verification
    # ------------------------------------------------------------------ #
    def verify(self, left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
        """So hai embedding với nhau. Không chạm vào index.

        Có hàm này để khái niệm verification là *code chạy được*, không phải chỉ là
        một đoạn giải thích trong README.
        """
        a = np.asarray(left, dtype=np.float32).ravel()
        b = np.asarray(right, dtype=np.float32).ravel()
        if a.shape != b.shape:
            raise ValidationFailed(
                f"hai vector khác chiều: {a.shape[0]} vs {b.shape[0]}",
                details={"left_dim": int(a.shape[0]), "right_dim": int(b.shape[0])},
            )
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            raise ValidationFailed("không so sánh được vector 0")
        score = float((a @ b) / (na * nb))
        threshold = self._config.match_threshold
        if score >= threshold:
            status = MATCH
        elif score >= threshold - self._config.review_band:
            status = NEEDS_REVIEW
        else:
            status = NO_MATCH
        return {
            "similarity_score": round(score, 6),
            "threshold": round(threshold, 6),
            "match_status": status,
            "threshold_calibrated": self._config.calibrated,
            "note": "cosine similarity, không phải xác suất cùng danh tính",
        }

    # ------------------------------------------------------------------ #
    # Đăng ký
    # ------------------------------------------------------------------ #
    def register_faces(
        self,
        faces: Sequence[DetectedFace],
        *,
        subject_id: str,
        principal: Principal,
        collection: str,
        embedding_space: str,
        metadata: dict[str, Any] | None = None,
        consent_ref: str | None = None,
        visibility: str = "collection",
    ) -> list[str]:
        """Đăng ký các khuôn mặt vào index dưới một ``subject_id``.

        ``consent_ref`` không bắt buộc ở tầng kiểu dữ liệu nhưng được log lại: đăng
        ký dữ liệu sinh trắc học mà không có tham chiếu tới cơ sở đồng ý là điều
        cần nhìn thấy được trong log, không phải điều lặng lẽ xảy ra.
        """
        principal.require_role(ROLE_REGISTER)
        if not subject_id:
            raise ValidationFailed("register cần subject_id")

        usable = [f for f in faces if f.is_usable and f.has_embedding]
        if not usable:
            raise ValidationFailed(
                "không có khuôn mặt nào dùng được để đăng ký",
                details={"faces_given": len(faces)},
            )

        now = time.time()
        records = [
            FaceRecord(
                face_id=f"face_{uuid.uuid4().hex[:16]}",
                subject_id=subject_id,
                embedding=face.embedding,  # type: ignore[arg-type]  # đã lọc has_embedding
                embedding_space=embedding_space,
                collection=collection,
                owner_id=principal.user_id,
                visibility=visibility,
                metadata={
                    **(metadata or {}),
                    "source_face_index": face.face_index,
                    "box": [round(v, 2) for v in face.box.as_list()],
                },
                created_at=now,
                consent_ref=consent_ref,
            )
            for face in usable
        ]

        if consent_ref is None:
            logger.warning(
                "face_register_without_consent_ref",
                extra={
                    "subject_id": subject_id,
                    "collection": collection,
                    "faces": len(records),
                    **principal.log_fields(),
                },
            )

        face_ids = self._repo.register(records, principal=principal)
        logger.info(
            "face_registered",
            extra={
                "subject_id": subject_id,
                "collection": collection,
                "faces": len(face_ids),
                "has_consent_ref": consent_ref is not None,
                **principal.log_fields(),
            },
        )
        return face_ids

    def stats(self, collection: str) -> dict[str, Any]:
        s = self._repo.stats(collection)
        return {
            "collection": s.collection,
            "total_faces": s.total_faces,
            "total_subjects": s.total_subjects,
            "dim": s.dim,
            "embedding_space": s.embedding_space,
            "metric": s.metric,
        }

    def delete_faces(self, face_ids: Sequence[str], *, principal: Principal) -> int:
        return self._repo.delete(face_ids, principal=principal)
