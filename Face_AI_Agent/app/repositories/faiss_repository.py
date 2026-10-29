"""Kho vector bằng FAISS - backend mặc định cho demo.

Vì sao FAISS cho bản này
------------------------
Yêu cầu là "chưa có database thì tự tạo tập test và tìm trong FAISS". FAISS chạy
in-process, không cần dịch vụ ngoài, và với vài nghìn vector thì ``IndexFlatIP`` cho
kết quả **chính xác tuyệt đối** (exhaustive search, không phải xấp xỉ) - đúng thứ
ta muốn khi đang hiệu chuẩn ngưỡng: sai số của ANN sẽ lẫn vào sai số của model.

Khi nào phải đổi: cần sharding/replica, cần filter theo metadata ở tầng engine, cần
nhiều process ghi đồng thời, hoặc vượt khoảng 10^6 vector. Lúc đó hiện thực
``BaseFaceSearchRepository`` bằng Elasticsearch kNN / pgvector; service và graph
không đổi một dòng.

Chọn metric
-----------
Backend trả vector **đã L2-normalize** (đo được ||v|| = 1.0). Với vector đơn vị thì
inner product = cosine similarity, nên ``IndexFlatIP`` cho đúng cosine mà không
phải chuẩn hoá thêm. Dùng ``IndexFlatL2`` cũng được (L2^2 = 2 - 2*cos) nhưng khi đó
ngưỡng phải nghịch đảo, dễ gây lỗi dấu - nên chốt cosine ở cả config và index.

Lọc theo metadata và phân quyền
-------------------------------
FAISS không filter được. Giải pháp: **over-fetch rồi lọc** - lấy nhiều hơn top_k
rồi bỏ những record principal không được đọc / không khớp metadata. Hệ quả cần biết:
nếu phần lớn index bị filter loại thì over-fetch có thể không đủ để lấp đủ top_k.
``_OVERSAMPLE`` xử lý trường hợp thường gặp; khi filter chọn lọc rất mạnh thì đó
chính là lúc nên chuyển sang backend filter được ở tầng engine.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.errors import VectorStoreError
from app.repositories.base import FaceRecord, RepositoryStats, SearchHit
from app.security import ROLE_DELETE, ROLE_READ_METADATA, ROLE_REGISTER, Principal

logger = logging.getLogger(__name__)

_OVERSAMPLE = 4
_OVERSAMPLE_BUFFER = 20
_METRIC = "cosine"


class _Collection:
    """Index + sidecar metadata của một collection."""

    __slots__ = ("id_of_face", "index", "name", "next_id", "records")

    def __init__(self, name: str, dim: int) -> None:
        self.name = name
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self.records: dict[int, FaceRecord] = {}
        self.id_of_face: dict[str, int] = {}
        self.next_id = 0


class FaissFaceRepository:
    """Hiện thực ``BaseFaceSearchRepository`` trên FAISS, có persistence ra đĩa."""

    def __init__(
        self,
        index_dir: str | Path,
        *,
        dim: int,
        embedding_space: str,
        autoload: bool = True,
    ) -> None:
        self._dir = Path(index_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._dim = dim
        self._embedding_space = embedding_space
        self._collections: dict[str, _Collection] = {}
        self._lock = threading.RLock()
        if autoload:
            for meta_path in sorted(self._dir.glob("*.meta.json")):
                name = meta_path.name[: -len(".meta.json")]
                try:
                    self._load_collection(name)
                except VectorStoreError:
                    raise
                except Exception as exc:  # pragma: no cover - file hỏng
                    raise VectorStoreError(
                        f"không đọc được collection {name!r}: {exc}",
                        details={"path": str(meta_path)},
                    ) from exc

    # ------------------------------------------------------------------ #
    # Thuộc tính
    # ------------------------------------------------------------------ #
    @property
    def dim(self) -> int:
        return self._dim

    @property
    def embedding_space(self) -> str:
        return self._embedding_space

    # ------------------------------------------------------------------ #
    # Ghi
    # ------------------------------------------------------------------ #
    def register(self, records: Sequence[FaceRecord], *, principal: Principal) -> list[str]:
        principal.require_role(ROLE_REGISTER)
        if not records:
            return []

        written: list[str] = []
        with self._lock:
            touched: set[str] = set()
            for record in records:
                self._check_record(record)
                col = self._get_or_create(record.collection)

                # Upsert: face_id đã tồn tại -> xoá vector cũ trước khi thêm mới.
                old_id = col.id_of_face.get(record.face_id)
                if old_id is not None:
                    col.index.remove_ids(np.array([old_id], dtype=np.int64))
                    col.records.pop(old_id, None)

                int_id = col.next_id
                col.next_id += 1
                vec = self._prepare_vector(record.embedding)
                col.index.add_with_ids(vec.reshape(1, -1), np.array([int_id], dtype=np.int64))

                if not record.created_at:
                    record.created_at = time.time()
                col.records[int_id] = record
                col.id_of_face[record.face_id] = int_id
                written.append(record.face_id)
                touched.add(record.collection)

            for name in touched:
                self._persist_collection(name)

        logger.info(
            "faiss_register",
            extra={"written": len(written), "collections": sorted({r.collection for r in records})},
        )
        return written

    def delete(self, face_ids: Sequence[str], *, principal: Principal) -> int:
        principal.require_role(ROLE_DELETE)
        removed = 0
        with self._lock:
            touched: set[str] = set()
            for face_id in face_ids:
                for name, col in self._collections.items():
                    int_id = col.id_of_face.get(face_id)
                    if int_id is None:
                        continue
                    record = col.records.get(int_id)
                    if record is not None and not principal.is_admin:
                        principal.require_owner(record.owner_id, what="face record")
                    col.index.remove_ids(np.array([int_id], dtype=np.int64))
                    col.records.pop(int_id, None)
                    col.id_of_face.pop(face_id, None)
                    removed += 1
                    touched.add(name)
            for name in touched:
                self._persist_collection(name)
        logger.info("faiss_delete", extra={"removed": removed})
        return removed

    # ------------------------------------------------------------------ #
    # Đọc
    # ------------------------------------------------------------------ #
    def search(
        self,
        vectors: np.ndarray,
        *,
        principal: Principal,
        top_k: int,
        collection: str,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[list[SearchHit]]:
        principal.require_collection(collection)

        queries = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        if queries.shape[1] != self._dim:
            raise VectorStoreError(
                f"vector truy vấn {queries.shape[1]}-D không khớp index {self._dim}-D",
                details={"query_dim": int(queries.shape[1]), "index_dim": self._dim},
            )
        if top_k <= 0:
            raise VectorStoreError("top_k phải > 0", details={"top_k": top_k})

        with self._lock:
            col = self._collections.get(collection)
            if col is None or col.index.ntotal == 0:
                # Index rỗng là trạng thái hợp lệ, không phải lỗi.
                return [[] for _ in range(queries.shape[0])]

            fetch = min(col.index.ntotal, top_k * _OVERSAMPLE + _OVERSAMPLE_BUFFER)
            scores, ids = col.index.search(queries, fetch)
            snapshot = dict(col.records)

        show_metadata = principal.has_role(ROLE_READ_METADATA)
        results: list[list[SearchHit]] = []

        # strict=True: FAISS luôn trả hai ma trận cùng shape; nếu một ngày không
        # còn vậy thì ta muốn biết ngay, không muốn lặng lẽ bỏ qua phần dư.
        for row_scores, row_ids in zip(scores, ids, strict=True):
            hits: list[SearchHit] = []
            for score, int_id in zip(row_scores, row_ids, strict=True):
                if int_id < 0:  # FAISS điền -1 khi không còn ứng viên
                    continue
                record = snapshot.get(int(int_id))
                if record is None:  # đã bị xoá giữa search và đọc sidecar
                    continue
                if not self._visible_to(record, principal):
                    continue
                if not self._matches_filter(record, metadata_filter):
                    continue
                hits.append(
                    SearchHit(
                        face_id=record.face_id,
                        subject_id=record.subject_id,
                        score=float(score),
                        rank=0,  # gán sau khi cắt top_k
                        collection=record.collection,
                        metadata=dict(record.metadata) if show_metadata else {},
                        metadata_visible=show_metadata,
                    )
                )
                if len(hits) >= top_k:
                    break
            for rank, hit in enumerate(hits, start=1):
                hit.rank = rank
            results.append(hits)

        return results

    def stats(self, collection: str) -> RepositoryStats:
        with self._lock:
            col = self._collections.get(collection)
            if col is None:
                return RepositoryStats(
                    collection=collection,
                    total_faces=0,
                    total_subjects=0,
                    dim=self._dim,
                    embedding_space=self._embedding_space,
                    metric=_METRIC,
                )
            return RepositoryStats(
                collection=collection,
                total_faces=int(col.index.ntotal),
                total_subjects=len({r.subject_id for r in col.records.values()}),
                dim=self._dim,
                embedding_space=self._embedding_space,
                metric=_METRIC,
            )

    def list_collections(self) -> list[str]:
        with self._lock:
            return sorted(self._collections)

    def health(self) -> dict[str, Any]:
        with self._lock:
            return {
                "backend": "faiss",
                "index_dir": str(self._dir),
                "metric": _METRIC,
                "dim": self._dim,
                "embedding_space": self._embedding_space,
                "collections": {
                    name: int(col.index.ntotal) for name, col in sorted(self._collections.items())
                },
            }

    # ------------------------------------------------------------------ #
    # Nội bộ
    # ------------------------------------------------------------------ #
    def _check_record(self, record: FaceRecord) -> None:
        if record.embedding_space != self._embedding_space:
            # Đây là bug im lặng nguy hiểm nhất của hệ thống này: trộn 'aligned' với
            # 'unaligned' cho similarity ~ 0.04 giữa hai ảnh CÙNG một người, nên mọi
            # kết quả search đều sai mà không có dấu hiệu nào. Chặn ở cửa.
            raise VectorStoreError(
                f"record {record.face_id} thuộc không gian {record.embedding_space!r} "
                f"nhưng index dùng {self._embedding_space!r}. Hai không gian này không "
                "so sánh được với nhau; hãy build lại index.",
                details={
                    "record_space": record.embedding_space,
                    "index_space": self._embedding_space,
                },
            )
        if not record.face_id or not record.subject_id:
            raise VectorStoreError(
                "record phải có cả face_id và subject_id",
                details={"face_id": record.face_id, "subject_id": record.subject_id},
            )

    def _prepare_vector(self, embedding: np.ndarray) -> np.ndarray:
        vec = np.asarray(embedding, dtype=np.float32).ravel()
        if vec.shape[0] != self._dim:
            raise VectorStoreError(
                f"vector {vec.shape[0]}-D không khớp index {self._dim}-D",
                details={"got": int(vec.shape[0]), "expected": self._dim},
            )
        if not np.all(np.isfinite(vec)):
            raise VectorStoreError("vector chứa NaN/Inf")
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            raise VectorStoreError("vector 0 không đăng ký được")
        # IndexFlatIP cho cosine chỉ khi vector là đơn vị -> chuẩn hoá tại cửa ghi.
        if abs(norm - 1.0) > 1e-5:
            vec = vec / norm
        return vec

    @staticmethod
    def _visible_to(record: FaceRecord, principal: Principal) -> bool:
        if principal.is_admin:
            return True
        if record.visibility == "private" and record.owner_id != principal.user_id:
            return False
        return principal.can_read_collection(record.collection)

    @staticmethod
    def _matches_filter(record: FaceRecord, flt: dict[str, Any] | None) -> bool:
        if not flt:
            return True
        for key, expected in flt.items():
            actual = record.metadata.get(key)
            if isinstance(expected, (list, tuple, set)):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True

    def _get_or_create(self, name: str) -> _Collection:
        col = self._collections.get(name)
        if col is None:
            col = _Collection(name, self._dim)
            self._collections[name] = col
        return col

    def _paths(self, name: str) -> tuple[Path, Path]:
        return self._dir / f"{name}.faiss", self._dir / f"{name}.meta.json"

    def _persist_collection(self, name: str) -> None:
        col = self._collections[name]
        index_path, meta_path = self._paths(name)
        faiss.write_index(col.index, str(index_path))
        meta = {
            "collection": name,
            "dim": self._dim,
            "embedding_space": self._embedding_space,
            "metric": _METRIC,
            "next_id": col.next_id,
            "saved_at": time.time(),
            "records": {str(k): v.to_sidecar() for k, v in col.records.items()},
        }
        tmp = meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(meta_path)  # ghi nguyên tử: không để lại file meta nửa vời

    def _load_collection(self, name: str) -> None:
        index_path, meta_path = self._paths(name)
        if not meta_path.exists():
            return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        # Index đã lưu phải khớp cấu hình hiện tại, nếu không thì mọi similarity đọc
        # ra đều vô nghĩa. Thà không khởi động được còn hơn chạy với kết quả sai.
        if int(meta.get("dim", -1)) != self._dim:
            raise VectorStoreError(
                f"collection {name!r} có dim {meta.get('dim')} nhưng config là {self._dim}",
                details={"path": str(meta_path)},
            )
        if meta.get("embedding_space") != self._embedding_space:
            raise VectorStoreError(
                f"collection {name!r} được build trong không gian "
                f"{meta.get('embedding_space')!r} nhưng config đang là "
                f"{self._embedding_space!r}. Hai không gian không so sánh được; "
                "hãy build lại index hoặc đổi vision.embedding_space về giá trị cũ.",
                details={"path": str(meta_path)},
            )

        col = _Collection(name, self._dim)
        if index_path.exists():
            col.index = faiss.read_index(str(index_path))
        col.next_id = int(meta.get("next_id", 0))
        for raw_id, sidecar in (meta.get("records") or {}).items():
            record = FaceRecord(
                face_id=sidecar["face_id"],
                subject_id=sidecar["subject_id"],
                embedding=np.zeros(0, dtype=np.float32),  # vector thật nằm trong index
                embedding_space=sidecar["embedding_space"],
                collection=sidecar.get("collection", name),
                owner_id=sidecar.get("owner_id", "system"),
                visibility=sidecar.get("visibility", "collection"),
                metadata=sidecar.get("metadata") or {},
                created_at=float(sidecar.get("created_at") or 0.0),
                consent_ref=sidecar.get("consent_ref"),
            )
            col.records[int(raw_id)] = record
            col.id_of_face[record.face_id] = int(raw_id)

        self._collections[name] = col
        logger.info(
            "faiss_collection_loaded",
            extra={"collection": name, "faces": int(col.index.ntotal)},
        )
