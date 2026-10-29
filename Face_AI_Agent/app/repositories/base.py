"""Interface cho kho vector khuôn mặt.

Business logic (``app/services``, ``app/agent``) chỉ biết tới interface này. Đổi từ
FAISS sang Elasticsearch kNN / Milvus / pgvector là viết một class mới hiện thực
``BaseFaceSearchRepository``, không sửa service hay node nào.

Hai quyết định đáng nói vì chúng nằm trong *interface*, không phải trong implementation:

1. ``search`` nhận **ma trận** ``(n_queries, dim)`` chứ không phải một vector.
   Một ảnh có nhiều mặt là trường hợp bình thường, và mọi backend vector đều tìm
   theo batch nhanh hơn nhiều so với gọi n lần. Nhờ vậy graph không cần fan-out
   song song cho từng mặt.
2. ``search`` nhận ``principal`` là tham số **bắt buộc**, không phải tuỳ chọn.
   Không thể viết một lời gọi search bỏ qua phân quyền.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from app.security import Principal


@dataclass(slots=True)
class FaceRecord:
    """Một khuôn mặt đã đăng ký trong kho."""

    face_id: str
    subject_id: str
    embedding: np.ndarray
    embedding_space: str
    collection: str = "default"
    owner_id: str = "system"
    visibility: str = "collection"  # "collection" | "private"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    # Tham chiếu tới bằng chứng đồng ý thu thập dữ liệu sinh trắc học.
    consent_ref: str | None = None

    def to_sidecar(self) -> dict[str, Any]:
        """Dạng JSON để lưu kèm index (không chứa embedding - nó nằm trong index)."""
        data = asdict(self)
        data.pop("embedding", None)
        return data


@dataclass(slots=True)
class SearchHit:
    """Một ứng viên trả về từ kho.

    ``score`` là **cosine similarity** trong [-1, 1], KHÔNG phải xác suất danh tính.
    Việc gán nhãn match / ambiguous / no_match là việc của ``FaceSearchService``,
    không phải của repository: repository chỉ biết khoảng cách.
    """

    face_id: str
    subject_id: str
    score: float
    rank: int
    collection: str
    metadata: dict[str, Any] = field(default_factory=dict)
    metadata_visible: bool = True


@dataclass(slots=True)
class RepositoryStats:
    collection: str
    total_faces: int
    total_subjects: int
    dim: int
    embedding_space: str
    metric: str


@runtime_checkable
class BaseFaceSearchRepository(Protocol):
    """Kho vector có phân quyền."""

    @property
    def dim(self) -> int: ...

    @property
    def embedding_space(self) -> str: ...

    def register(self, records: Sequence[FaceRecord], *, principal: Principal) -> list[str]:
        """Thêm/ghi đè các record. Trả về danh sách ``face_id`` đã ghi."""
        ...

    def search(
        self,
        vectors: np.ndarray,
        *,
        principal: Principal,
        top_k: int,
        collection: str,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[list[SearchHit]]:
        """Tìm top-k cho từng vector truy vấn.

        Trả về list song song với ``vectors``: phần tử thứ i là kết quả của vector i.
        Index rỗng -> trả list các list rỗng (KHÔNG raise: "chưa có dữ liệu" là một
        câu trả lời hợp lệ mà người dùng cần biết).
        """
        ...

    def delete(self, face_ids: Sequence[str], *, principal: Principal) -> int:
        """Xoá theo chính sách lưu trữ / yêu cầu thu hồi. Trả số record đã xoá."""
        ...

    def stats(self, collection: str) -> RepositoryStats: ...

    def health(self) -> dict[str, Any]: ...
