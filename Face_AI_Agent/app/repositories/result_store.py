"""Lưu kết quả detection/search có cấu trúc, tách khỏi lịch sử hội thoại.

Vì sao không dùng conversation history
--------------------------------------
Khi người dùng hỏi tiếp "kết quả thứ hai có similarity bao nhiêu?", câu trả lời
**không được** lấy từ text mà LLM đã viết ra ở lượt trước. Lý do:

* Text là thứ LLM sinh ra; đọc lại nó rồi trả lời là nhân bản sai số của chính mình.
* Checkpoint của LangGraph lưu state theo thread, nhưng nó phục vụ *resume workflow*,
  không phải *tra cứu kết quả có phân quyền*. Hai mục đích khác nhau thì hai kho
  khác nhau.
* Kết quả sinh trắc học cần TTL và quyền truy cập riêng; checkpoint thì không có
  khái niệm đó.

Nên: kết quả có cấu trúc nằm ở đây, kèm ``owner_id``, ``thread_id`` và ``expires_at``.
Node ``answer_question`` đọc từ kho này, không đọc từ messages.

Ba loại ID, đừng lẫn
--------------------
* ``thread_id``  - một cuộc hội thoại. Dùng làm khoá checkpoint của LangGraph.
* ``request_id`` - một lần gọi API. Dùng để trace/log; mỗi lượt chat một request_id.
* ``result_id``  - một kết quả detection hoặc search cụ thể. Đây là thứ được tham
  chiếu khi hỏi tiếp, và là thứ có ACL + TTL.

Embedding được lưu ở bảng riêng dưới dạng BLOB float32, không nằm trong payload JSON:
payload là thứ đi ra API và đi vào prompt, nên nó phải *không bao giờ* chứa vector.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.errors import ResultNotFound
from app.logging_utils import new_result_id
from app.security import Principal

logger = logging.getLogger(__name__)

KIND_DETECTION = "detection"
KIND_SEARCH = "search"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    result_id   TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    owner_id    TEXT NOT NULL,
    thread_id   TEXT,
    request_id  TEXT,
    created_at  REAL NOT NULL,
    expires_at  REAL NOT NULL,
    payload     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_results_thread ON results (thread_id, kind, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_results_owner  ON results (owner_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_results_expiry ON results (expires_at);

CREATE TABLE IF NOT EXISTS result_embeddings (
    result_id       TEXT NOT NULL,
    face_index      INTEGER NOT NULL,
    dim             INTEGER NOT NULL,
    embedding_space TEXT NOT NULL,
    vector          BLOB NOT NULL,
    PRIMARY KEY (result_id, face_index)
);
"""


@dataclass(slots=True)
class StoredResult:
    result_id: str
    kind: str
    owner_id: str
    thread_id: str | None
    request_id: str | None
    created_at: float
    expires_at: float
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.created_at)


class SqliteResultStore:
    """Kho kết quả trên SQLite.

    Giới hạn đã biết (demo): SQLite ghi tuần tự, một writer tại một thời điểm, và
    không chia sẻ được giữa nhiều máy. Với nhiều replica hoặc tải ghi cao, đổi sang
    PostgreSQL - interface của class này không đổi, chỉ thay phần kết nối và cú pháp
    upsert. WAL được bật để đọc không bị chặn bởi ghi.
    """

    def __init__(self, db_path: str | Path, *, default_ttl_seconds: int = 86400) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ttl = default_ttl_seconds
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        # Một connection cho mỗi thao tác: endpoint FastAPI sync chạy trong threadpool,
        # mà connection của sqlite3 không an toàn khi dùng chéo thread.
        conn = sqlite3.connect(self._path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    # ------------------------------------------------------------------ #
    # Ghi
    # ------------------------------------------------------------------ #
    def save(
        self,
        *,
        kind: str,
        owner_id: str,
        payload: dict[str, Any],
        thread_id: str | None = None,
        request_id: str | None = None,
        embeddings: dict[int, np.ndarray] | None = None,
        embedding_space: str | None = None,
        ttl_seconds: int | None = None,
    ) -> str:
        if kind not in (KIND_DETECTION, KIND_SEARCH):
            raise ValueError(f"kind không hợp lệ: {kind!r}")

        result_id = new_result_id(kind[:3])
        now = time.time()
        ttl = self._ttl if ttl_seconds is None else ttl_seconds

        rows: list[tuple[Any, ...]] = []
        if embeddings:
            if not embedding_space:
                raise ValueError("lưu embedding thì phải nói rõ embedding_space")
            for face_index, vec in embeddings.items():
                arr = np.ascontiguousarray(np.asarray(vec, dtype=np.float32).ravel())
                rows.append((result_id, int(face_index), int(arr.shape[0]), embedding_space, arr.tobytes()))

        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO results (result_id, kind, owner_id, thread_id, request_id, "
                "created_at, expires_at, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    result_id,
                    kind,
                    owner_id,
                    thread_id,
                    request_id,
                    now,
                    now + ttl,
                    json.dumps(payload, ensure_ascii=False, default=str),
                ),
            )
            if rows:
                conn.executemany(
                    "INSERT INTO result_embeddings (result_id, face_index, dim, embedding_space, vector) "
                    "VALUES (?, ?, ?, ?, ?)",
                    rows,
                )
            conn.commit()

        logger.info(
            "result_saved",
            extra={
                "result_id": result_id,
                "kind": kind,
                "thread_id": thread_id,
                "request_id": request_id,
                "embeddings_stored": len(rows),
            },
        )
        return result_id

    # ------------------------------------------------------------------ #
    # Đọc - luôn qua principal
    # ------------------------------------------------------------------ #
    def get(self, result_id: str, *, principal: Principal, kind: str | None = None) -> StoredResult:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM results WHERE result_id = ?", (result_id,)
            ).fetchone()

        if row is None:
            raise ResultNotFound(f"không có kết quả {result_id!r}")
        if kind is not None and row["kind"] != kind:
            raise ResultNotFound(f"kết quả {result_id!r} không phải loại {kind!r}")
        if row["expires_at"] < time.time():
            # Hết hạn = coi như không tồn tại. Dọn luôn cho khỏi tích tụ.
            self.delete(result_id)
            raise ResultNotFound(f"kết quả {result_id!r} đã hết hạn lưu trữ")

        # Chốt phân quyền. Thông báo lỗi không tiết lộ gì về nội dung kết quả.
        principal.require_owner(row["owner_id"], what="result")

        return self._to_result(row)

    def latest(
        self,
        *,
        kind: str,
        principal: Principal,
        thread_id: str | None = None,
    ) -> StoredResult | None:
        """Kết quả mới nhất còn hiệu lực *của chính principal này*.

        Việc lọc ``owner_id`` nằm ngay trong câu SQL, không phải kiểm tra sau: như vậy
        một thread_id của người khác cũng không thể lọt ra, kể cả khi caller truyền
        thread_id đoán được.
        """
        sql = (
            "SELECT * FROM results WHERE kind = ? AND expires_at > ? "
            "{owner} {thread} ORDER BY created_at DESC LIMIT 1"
        )
        params: list[Any] = [kind, time.time()]
        owner_clause = ""
        if not principal.is_admin:
            owner_clause = "AND owner_id = ?"
            params.append(principal.user_id)
        thread_clause = ""
        if thread_id:
            thread_clause = "AND thread_id = ?"
            params.append(thread_id)

        with self._connect() as conn:
            row = conn.execute(
                sql.format(owner=owner_clause, thread=thread_clause), tuple(params)
            ).fetchone()
        return self._to_result(row) if row else None

    def get_embeddings(
        self, result_id: str, *, principal: Principal
    ) -> tuple[dict[int, np.ndarray], str | None]:
        """Trả ``{face_index: vector}`` và embedding_space. ACL kiểm tra trước."""
        self.get(result_id, principal=principal)  # raise nếu không có quyền / hết hạn
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT face_index, dim, embedding_space, vector FROM result_embeddings "
                "WHERE result_id = ? ORDER BY face_index",
                (result_id,),
            ).fetchall()

        out: dict[int, np.ndarray] = {}
        space: str | None = None
        for row in rows:
            vec = np.frombuffer(row["vector"], dtype=np.float32)
            if vec.shape[0] != row["dim"]:  # pragma: no cover - DB bị sửa tay
                raise ResultNotFound(
                    f"embedding của {result_id} face #{row['face_index']} bị hỏng"
                )
            out[int(row["face_index"])] = vec.copy()  # copy: frombuffer là read-only view
            space = row["embedding_space"]
        return out, space

    def list_for_thread(
        self, *, thread_id: str, principal: Principal, limit: int = 20
    ) -> list[StoredResult]:
        params: list[Any] = [thread_id, time.time()]
        owner_clause = ""
        if not principal.is_admin:
            owner_clause = "AND owner_id = ?"
            params.append(principal.user_id)
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM results WHERE thread_id = ? AND expires_at > ? {owner_clause} "
                "ORDER BY created_at DESC LIMIT ?",
                tuple(params),
            ).fetchall()
        return [self._to_result(r) for r in rows]

    def list_for_request(
        self, *, request_id: str, principal: Principal, limit: int = 20
    ) -> list[StoredResult]:
        """Các kết quả gắn với một ``request_id``.

        Phục vụ ``GET /tasks/{request_id}``: client mất response vẫn tra lại được.
        Lọc owner nằm trong SQL, nên request_id của người khác không trả ra gì.
        """
        params: list[Any] = [request_id, time.time()]
        owner_clause = ""
        if not principal.is_admin:
            owner_clause = "AND owner_id = ?"
            params.append(principal.user_id)
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM results WHERE request_id = ? AND expires_at > ? {owner_clause} "
                "ORDER BY created_at ASC LIMIT ?",
                tuple(params),
            ).fetchall()
        return [self._to_result(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Xoá / chính sách lưu trữ
    # ------------------------------------------------------------------ #
    def delete(self, result_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM result_embeddings WHERE result_id = ?", (result_id,))
            conn.execute("DELETE FROM results WHERE result_id = ?", (result_id,))
            conn.commit()

    def delete_for_owner(self, owner_id: str, *, principal: Principal) -> int:
        """Thực thi quyền xoá dữ liệu của chủ thể (right to erasure)."""
        principal.require_owner(owner_id, what="result set")
        with self._lock, self._connect() as conn:
            ids = [
                r["result_id"]
                for r in conn.execute(
                    "SELECT result_id FROM results WHERE owner_id = ?", (owner_id,)
                ).fetchall()
            ]
            conn.executemany("DELETE FROM result_embeddings WHERE result_id = ?", [(i,) for i in ids])
            conn.execute("DELETE FROM results WHERE owner_id = ?", (owner_id,))
            conn.commit()
        logger.info("results_deleted_for_owner", extra={"deleted": len(ids)})
        return len(ids)

    def purge_expired(self, *, now: float | None = None) -> int:
        """Dọn kết quả hết hạn. Gọi định kỳ (startup + background job)."""
        cutoff = time.time() if now is None else now
        with self._lock, self._connect() as conn:
            ids = [
                r["result_id"]
                for r in conn.execute(
                    "SELECT result_id FROM results WHERE expires_at <= ?", (cutoff,)
                ).fetchall()
            ]
            if ids:
                conn.executemany(
                    "DELETE FROM result_embeddings WHERE result_id = ?", [(i,) for i in ids]
                )
                conn.execute("DELETE FROM results WHERE expires_at <= ?", (cutoff,))
                conn.commit()
        if ids:
            logger.info("results_purged", extra={"purged": len(ids)})
        return len(ids)

    def health(self) -> dict[str, Any]:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) AS n FROM results").fetchone()["n"]
            live = conn.execute(
                "SELECT COUNT(*) AS n FROM results WHERE expires_at > ?", (time.time(),)
            ).fetchone()["n"]
        return {
            "backend": "sqlite",
            "path": str(self._path),
            "results_total": int(total),
            "results_live": int(live),
            "default_ttl_seconds": self._ttl,
        }

    @staticmethod
    def _to_result(row: sqlite3.Row) -> StoredResult:
        return StoredResult(
            result_id=row["result_id"],
            kind=row["kind"],
            owner_id=row["owner_id"],
            thread_id=row["thread_id"],
            request_id=row["request_id"],
            created_at=float(row["created_at"]),
            expires_at=float(row["expires_at"]),
            payload=json.loads(row["payload"]),
        )


def summarize_results(results: Iterable[StoredResult]) -> list[dict[str, Any]]:
    """Dạng gọn để log / debug, không chứa payload."""
    return [
        {
            "result_id": r.result_id,
            "kind": r.kind,
            "thread_id": r.thread_id,
            "age_seconds": round(r.age_seconds, 1),
        }
        for r in results
    ]
