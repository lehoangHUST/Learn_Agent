"""Face Embedding Service.

Backend imintv3 trả embedding ngay trong lời gọi detect, nên nhiệm vụ chính của
service này không phải "chạy thêm một model" mà là **bảo đảm chất lượng vector
trước khi đem đi so khớp**:

* chỉ gọi ``embedder.embed()`` khi detector chưa cung cấp vector (backend khác),
* loại mặt quá nhỏ / thiếu vector,
* kiểm tra dim và không gian embedding khớp cấu hình,
* L2-normalize nếu cần,
* chọn đúng (các) mặt mà kế hoạch yêu cầu, ví dụ "khuôn mặt thứ hai".

Vì sao kiểm tra không gian embedding ở đây nữa, khi repository cũng kiểm tra:
đây là tầng duy nhất nhìn thấy *cả* vector truy vấn và cấu hình, nên nó bắt được
trường hợp đổi ``vision.embedding_space`` mà quên build lại index - trước khi câu
truy vấn chạm vào FAISS và trả về những con số trông hợp lệ nhưng vô nghĩa.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.config import VisionConfig
from app.errors import ValidationFailed
from app.logging_utils import timed
from app.vision.base import BaseFaceEmbedder, DetectedFace, FaceAnalysis

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EmbeddingBatch:
    """Ma trận vector truy vấn + ánh xạ về ``face_index`` gốc."""

    vectors: np.ndarray  # shape (n, dim), float32, L2-normalized
    face_indices: list[int]
    embedding_space: str
    skipped: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return self.vectors.size == 0

    @property
    def count(self) -> int:
        return len(self.face_indices)


class FaceEmbeddingService:
    def __init__(self, embedder: BaseFaceEmbedder, config: VisionConfig) -> None:
        self._embedder = embedder
        self._config = config

    @property
    def embedding_space(self) -> str:
        return self._config.embedding_space

    @property
    def embedding_dim(self) -> int:
        return self._config.embedding_dim

    def ensure_embeddings(self, analysis: FaceAnalysis, image_bytes: bytes) -> FaceAnalysis:
        """Bảo đảm mọi mặt dùng được đều có vector.

        Với imintv3 thì detector đã trả vector sẵn -> hàm này **không** gọi thêm
        HTTP, chỉ kiểm tra. Với một backend detector-only trong tương lai, nó mới
        gọi ``embed()``. Nhờ vậy số lời gọi backend không phụ thuộc vào số node
        trong graph.
        """
        missing = [f for f in analysis.faces if f.is_usable and not f.has_embedding]
        if not missing:
            return analysis

        with timed(logger, "face_embed", faces_missing=len(missing)) as extra:
            fresh = self._embedder.embed(image_bytes)
            extra["faces_returned"] = fresh.face_count

        # Ghép theo face_index sau khi cả hai đã qua order_faces -> thứ tự nhất quán.
        by_index = {f.face_index: f for f in fresh.faces}
        for face in missing:
            source = by_index.get(face.face_index)
            if source is None or source.embedding is None:
                face.rejected_reason = "embedding_unavailable"
                analysis.warnings.append(
                    f"face #{face.face_index}: embedder không trả vector, bị loại khỏi search"
                )
                continue
            face.embedding = source.embedding
            face.embedding_space = source.embedding_space
        return analysis

    def select_faces(
        self,
        analysis: FaceAnalysis,
        face_indices: Sequence[int] | None = None,
    ) -> tuple[list[DetectedFace], list[dict[str, Any]]]:
        """Chọn các mặt sẽ đem đi search.

        ``face_indices=None`` -> mọi mặt dùng được. Ngược lại chỉ những index được
        yêu cầu (dùng cho "tìm khuôn mặt thứ hai"). Index ngoài phạm vi là lỗi đầu
        vào, không im lặng bỏ qua: người dùng cần biết ảnh chỉ có 3 mặt khi họ hỏi
        về mặt thứ 5.
        """
        skipped: list[dict[str, Any]] = []

        if face_indices is None:
            chosen = list(analysis.faces)
        else:
            valid = {f.face_index for f in analysis.faces}
            unknown = [i for i in face_indices if i not in valid]
            if unknown:
                raise ValidationFailed(
                    f"ảnh chỉ có {analysis.face_count} khuôn mặt (index 0..{analysis.face_count - 1}); "
                    f"không có index {unknown}",
                    details={"requested": list(face_indices), "available": sorted(valid)},
                )
            by_index = {f.face_index: f for f in analysis.faces}
            chosen = [by_index[i] for i in face_indices]

        usable: list[DetectedFace] = []
        for face in chosen:
            if not face.is_usable:
                skipped.append({"face_index": face.face_index, "reason": face.rejected_reason})
                continue
            if not face.has_embedding:
                skipped.append({"face_index": face.face_index, "reason": "missing_embedding"})
                continue
            usable.append(face)
        return usable, skipped

    def build_matrix(self, faces: Sequence[DetectedFace]) -> EmbeddingBatch:
        """Gom vector của các mặt thành một ma trận để search theo batch."""
        if not faces:
            return EmbeddingBatch(
                vectors=np.zeros((0, self.embedding_dim), dtype=np.float32),
                face_indices=[],
                embedding_space=self.embedding_space,
            )

        rows: list[np.ndarray] = []
        indices: list[int] = []
        skipped: list[dict[str, Any]] = []

        for face in faces:
            vec = face.embedding
            if vec is None:
                skipped.append({"face_index": face.face_index, "reason": "missing_embedding"})
                continue
            if face.embedding_space and face.embedding_space != self.embedding_space:
                # Chặn sớm: vector của không gian khác sẽ cho similarity ~ 0 với mọi
                # thứ trong index, trông như "không tìm thấy ai" thay vì như một bug.
                raise ValidationFailed(
                    f"face #{face.face_index} có vector thuộc không gian "
                    f"{face.embedding_space!r} nhưng hệ thống đang dùng "
                    f"{self.embedding_space!r}",
                    details={
                        "face_space": face.embedding_space,
                        "configured_space": self.embedding_space,
                    },
                )
            arr = np.asarray(vec, dtype=np.float32).ravel()
            if arr.shape[0] != self.embedding_dim:
                raise ValidationFailed(
                    f"face #{face.face_index} có vector {arr.shape[0]}-D, cần {self.embedding_dim}-D",
                    details={"got": int(arr.shape[0]), "expected": self.embedding_dim},
                )
            norm = float(np.linalg.norm(arr))
            if norm == 0.0 or not np.isfinite(norm):
                skipped.append({"face_index": face.face_index, "reason": "degenerate_vector"})
                continue
            if self._config.enforce_l2_normalize and abs(norm - 1.0) > 1e-5:
                arr = arr / norm
            rows.append(arr)
            indices.append(face.face_index)

        vectors = (
            np.vstack(rows).astype(np.float32)
            if rows
            else np.zeros((0, self.embedding_dim), dtype=np.float32)
        )
        return EmbeddingBatch(
            vectors=vectors,
            face_indices=indices,
            embedding_space=self.embedding_space,
            skipped=skipped,
        )

    def build_matrix_from_stored(
        self,
        stored: dict[int, np.ndarray],
        *,
        embedding_space: str | None,
        face_indices: Sequence[int] | None = None,
    ) -> EmbeddingBatch:
        """Dựng ma trận truy vấn từ embedding đã lưu trong result store.

        Đây là đường đi dùng chung cho **cả hai** tình huống: tìm kiếm ngay sau khi
        detect ở lượt này, và tìm kiếm dựa trên kết quả detect của lượt trước
        ("tìm khuôn mặt thứ hai"). Cả hai đọc cùng một nguồn nên không thể lệch logic
        giữa lượt đầu và lượt sau - đó là loại bug rất khó thấy nếu viết hai đường
        riêng.

        ``face_indices`` dùng chỉ số **global** như trong payload detection: với một
        ảnh thì nó trùng ``face_index``, với nhiều ảnh thì nó là số thứ tự chạy suốt
        các ảnh.
        """
        if embedding_space is not None and embedding_space != self.embedding_space:
            raise ValidationFailed(
                f"embedding đã lưu thuộc không gian {embedding_space!r} nhưng hệ thống "
                f"đang dùng {self.embedding_space!r}; kết quả cũ không còn so khớp được",
                details={"stored_space": embedding_space, "configured_space": self.embedding_space},
            )

        available = sorted(stored)
        if face_indices is None:
            wanted = available
        else:
            unknown = [i for i in face_indices if i not in stored]
            if unknown:
                raise ValidationFailed(
                    f"kết quả này có {len(available)} khuôn mặt dùng được "
                    f"(index {available}); không có index {unknown}",
                    details={"requested": list(face_indices), "available": available},
                )
            wanted = list(face_indices)

        rows: list[np.ndarray] = []
        indices: list[int] = []
        skipped: list[dict[str, Any]] = []

        for idx in wanted:
            arr = np.asarray(stored[idx], dtype=np.float32).ravel()
            if arr.shape[0] != self.embedding_dim:
                raise ValidationFailed(
                    f"embedding đã lưu của face #{idx} có {arr.shape[0]} chiều, "
                    f"cần {self.embedding_dim}",
                    details={"got": int(arr.shape[0]), "expected": self.embedding_dim},
                )
            norm = float(np.linalg.norm(arr))
            if norm == 0.0 or not np.isfinite(norm):
                skipped.append({"face_index": idx, "reason": "degenerate_vector"})
                continue
            if self._config.enforce_l2_normalize and abs(norm - 1.0) > 1e-5:
                arr = arr / norm
            rows.append(arr)
            indices.append(idx)

        vectors = (
            np.vstack(rows).astype(np.float32)
            if rows
            else np.zeros((0, self.embedding_dim), dtype=np.float32)
        )
        return EmbeddingBatch(
            vectors=vectors,
            face_indices=indices,
            embedding_space=self.embedding_space,
            skipped=skipped,
        )
