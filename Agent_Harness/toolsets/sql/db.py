"""Truy cập SQLite chỉ-đọc, có giới hạn tài nguyên.

Hai thứ mà một DB client bình thường không cần nhưng harness thì bắt buộc:

* **Chỉ-đọc ở tầng engine** - URI ``mode=ro``. Guard ở ``guard.py`` có thể bỏ sót
  một dạng câu lệnh nào đó; lớp này thì không, vì SQLite từ chối ghi bất kể nội
  dung câu lệnh.
* **Timeout cắt được câu truy vấn đang chạy** - qua ``set_progress_handler``.
  ``sqlite3`` không có tham số timeout cho câu truy vấn (tham số ``timeout`` của
  ``connect`` là thời gian chờ *lock*, không liên quan). Mà một ``SELECT`` hoàn
  toàn hợp lệ do model viết - ví dụ join ba bảng không điều kiện - có thể chạy
  hàng phút. Progress handler được gọi định kỳ trong lúc thực thi và chỉ cần trả về
  khác 0 là SQLite huỷ câu truy vấn. Đã kiểm chứng bằng cách chạy thật.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.errors import ToolFailure
from toolsets.sql.guard import apply_limit, ensure_read_only

# Số lệnh ảo của VM SQLite giữa hai lần gọi progress handler. Nhỏ thì phản hồi
# nhanh nhưng tốn overhead; 1000 là cân bằng đủ cho một câu truy vấn cỡ giây.
_PROGRESS_INSTRUCTIONS = 1000


@dataclass(slots=True)
class SqlLimits:
    max_rows: int = 200
    max_query_seconds: float = 5.0
    max_cell_chars: int = 200


@dataclass(slots=True)
class QueryResult:
    columns: list[str]
    rows: list[tuple[Any, ...]]
    truncated_rows: bool
    duration_ms: float
    sql_executed: str
    limit_wrapped: bool
    # Giá trị quá dài bị cắt ở từng ô - một cột TEXT chứa cả bài viết sẽ làm nổ
    # context trước khi harness kịp cắt ở mức kết quả.
    truncated_cells: int = 0

    def to_markdown(self) -> str:
        if not self.columns:
            return "(truy vấn không trả về cột nào)"
        header = "| " + " | ".join(self.columns) + " |"
        divider = "| " + " | ".join("---" for _ in self.columns) + " |"
        body = [
            "| " + " | ".join("NULL" if v is None else str(v) for v in row) + " |"
            for row in self.rows
        ]
        lines = [header, divider, *body]
        lines.append("")
        lines.append(f"({len(self.rows)} dòng, {self.duration_ms:.0f}ms)")
        if self.truncated_rows:
            lines.append(
                f"LƯU Ý: kết quả bị cắt ở {len(self.rows)} dòng. Đây KHÔNG phải toàn bộ "
                "dữ liệu - dùng COUNT(*) hoặc GROUP BY nếu cần con số tổng."
            )
        if self.truncated_cells:
            lines.append(f"LƯU Ý: {self.truncated_cells} ô có giá trị dài bị cắt bớt.")
        if self.limit_wrapped:
            lines.append("(harness tự thêm LIMIT vì truy vấn không có LIMIT)")
        return "\n".join(lines)


@dataclass(slots=True)
class ReadOnlyDatabase:
    """Mỗi lần gọi mở một connection mới.

    Không dùng chung connection giữa các lượt: handler tool có thể được gọi từ
    nhiều thread (endpoint FastAPI sync chạy trong threadpool), mà connection của
    ``sqlite3`` không an toàn khi dùng chéo thread. Mở connection tới một file
    SQLite cục bộ là thao tác rẻ.
    """

    path: Path
    limits: SqlLimits = field(default_factory=SqlLimits)

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def _connect(self) -> sqlite3.Connection:
        if not self.path.exists():
            raise ToolFailure(
                f"không tìm thấy database tại {self.path}",
                code="database_missing",
                recoverable=False,
                hint="Chạy `python -m toolsets.sql.seed` để tạo database demo.",
            )
        conn = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)
        conn.row_factory = None
        return conn

    def _install_timeout(self, conn: sqlite3.Connection) -> None:
        deadline = time.monotonic() + self.limits.max_query_seconds

        def abort_if_slow() -> int:
            # Trả về khác 0 -> SQLite huỷ câu truy vấn và raise OperationalError.
            return 1 if time.monotonic() > deadline else 0

        conn.set_progress_handler(abort_if_slow, _PROGRESS_INSTRUCTIONS)

    # ------------------------------------------------------------------ #
    def list_tables(self) -> list[tuple[str, str]]:
        """``(tên, loại)`` của mọi bảng/view do người dùng tạo."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT name, type FROM sqlite_master "
                "WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' "
                "ORDER BY type, name"
            ).fetchall()
            return [(str(r[0]), str(r[1])) for r in rows]
        finally:
            conn.close()

    def describe_table(self, table: str) -> dict[str, Any]:
        """Schema của một bảng, kèm số dòng và vài dòng mẫu."""
        known = {name for name, _ in self.list_tables()}
        if table not in known:
            raise ToolFailure(
                f"không có bảng hay view nào tên {table!r}",
                code="unknown_table",
                recoverable=True,
                hint=f"Các bảng có sẵn: {', '.join(sorted(known))}",
            )

        conn = self._connect()
        try:
            self._install_timeout(conn)
            # table_info là PRAGMA, nhưng ở đây *code của ta* gọi với tên bảng đã
            # được kiểm trong allowlist - khác hoàn toàn với việc để SQL của model
            # chứa PRAGMA.
            columns = [
                {
                    "name": str(row[1]),
                    "type": str(row[2]) or "?",
                    "notnull": bool(row[3]),
                    "default": row[4],
                    "pk": bool(row[5]),
                }
                for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall()
            ]
            row_count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            return {"table": table, "columns": columns, "row_count": int(row_count)}
        except sqlite3.OperationalError as exc:
            raise self._as_tool_failure(exc) from exc
        finally:
            conn.close()

    def run_select(self, sql: str) -> QueryResult:
        """Chạy một câu đọc, sau khi đã qua guard và đã áp LIMIT."""
        statement = ensure_read_only(sql)
        # Bọc với max_rows + 1, không phải max_rows: nếu bọc đúng max_rows thì cursor
        # không bao giờ trả về quá max_rows dòng, nên phép kiểm "có bị cắt không"
        # (``len(rows) > max_rows``) luôn sai - và cảnh báo "đây không phải toàn bộ
        # dữ liệu" không bao giờ đến được model, đúng với những truy vấn dễ bị cắt
        # nhất. Lấy thêm một dòng là cách rẻ nhất để biết còn dữ liệu phía sau.
        executed, wrapped = apply_limit(statement, self.limits.max_rows + 1)

        conn = self._connect()
        started = time.perf_counter()
        try:
            self._install_timeout(conn)
            cursor = conn.execute(executed)
            columns = [str(d[0]) for d in (cursor.description or [])]
            # Lấy max_rows + 1 để biết có bị cắt hay không mà không phải đọc hết bảng.
            raw_rows = cursor.fetchmany(self.limits.max_rows + 1)
        except sqlite3.OperationalError as exc:
            raise self._as_tool_failure(exc) from exc
        except sqlite3.DatabaseError as exc:
            raise ToolFailure(
                f"SQLite từ chối truy vấn: {exc}",
                code="sql_error",
                recoverable=True,
                hint="Kiểm tra lại cú pháp và tên cột (dùng describe_table).",
            ) from exc
        finally:
            conn.close()

        duration = (time.perf_counter() - started) * 1000
        truncated_rows = len(raw_rows) > self.limits.max_rows
        rows = raw_rows[: self.limits.max_rows]

        clipped = 0
        safe_rows: list[tuple[Any, ...]] = []
        for row in rows:
            cells = []
            for value in row:
                if isinstance(value, str) and len(value) > self.limits.max_cell_chars:
                    cells.append(value[: self.limits.max_cell_chars] + "…[cắt]")
                    clipped += 1
                elif isinstance(value, bytes):
                    cells.append(f"<blob {len(value)} bytes>")
                else:
                    cells.append(value)
            safe_rows.append(tuple(cells))

        return QueryResult(
            columns=columns,
            rows=safe_rows,
            truncated_rows=truncated_rows,
            duration_ms=duration,
            sql_executed=executed,
            limit_wrapped=wrapped,
            truncated_cells=clipped,
        )

    def _as_tool_failure(self, exc: sqlite3.OperationalError) -> ToolFailure:
        text = str(exc)
        if "interrupted" in text:
            return ToolFailure(
                f"truy vấn bị huỷ vì chạy quá {self.limits.max_query_seconds}s",
                code="query_timeout",
                recoverable=True,
                hint=(
                    "Truy vấn quá nặng. Thêm WHERE để lọc, hoặc dùng COUNT/GROUP BY "
                    "thay vì lấy toàn bộ dòng, hoặc tránh join không có điều kiện."
                ),
            )
        if "readonly" in text:
            return ToolFailure(
                "database này là chỉ-đọc",
                code="not_read_only",
                recoverable=False,
                hint="Chỉ dùng câu SELECT.",
            )
        if "no such table" in text or "no such column" in text:
            return ToolFailure(
                f"{text}",
                code="unknown_object",
                recoverable=True,
                hint="Dùng list_tables và describe_table để lấy tên đúng.",
            )
        return ToolFailure(
            f"SQLite lỗi: {text}",
            code="sql_error",
            recoverable=True,
            hint="Kiểm tra lại cú pháp truy vấn.",
        )
