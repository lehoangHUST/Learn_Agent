"""Toolset SQL - phần "domain" ghép vào harness.

Năm tool, và sự phân loại của chúng là phần đáng chú ý nhất:

==================  =========  ==============  ====================================
tool                side_effect  cần phê duyệt  vì sao
==================  =========  ==============  ====================================
``list_tables``     read       không           chỉ đọc metadata
``describe_table``  read       không           chỉ đọc schema
``run_query``       read       không           engine chỉ-đọc + guard + LIMIT + timeout
``profile_column``  read       không           như trên, chỉ là tổng hợp sẵn
``save_finding``    **write**  **có**          ghi ra ngoài -> cần người đồng ý
==================  =========  ==============  ====================================

``save_finding`` tồn tại không phải vì một agent phân tích dữ liệu nhất thiết cần
nó, mà vì **một harness cần ít nhất một hành động có hậu quả để cổng phê duyệt là
thật**. Một harness mà mọi tool đều chỉ đọc thì phần human-in-the-loop chưa bao giờ
được kiểm chứng.

``run_query`` đòi thêm tham số ``purpose``: model phải viết ra nó đang tìm gì. Hai
lợi ích thật: transcript đọc được bởi người, và nó buộc model phát biểu ý định
trước khi viết SQL - thường làm SQL tốt hơn.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from harness.errors import ToolFailure
from harness.registry import (
    SIDE_EFFECT_READ,
    SIDE_EFFECT_WRITE,
    ToolRegistry,
    ToolRuntime,
    ToolSpec,
)
from toolsets.sql.db import ReadOnlyDatabase

PERM_SQL_READ = "sql:read"
PERM_FINDINGS_WRITE = "findings:write"


def _database(runtime: ToolRuntime) -> ReadOnlyDatabase:
    db = runtime.context.get("database")
    if not isinstance(db, ReadOnlyDatabase):  # pragma: no cover - lắp sai context
        raise ToolFailure(
            "harness chưa được cấu hình database",
            code="misconfigured",
            recoverable=False,
        )
    return db


# --------------------------------------------------------------------------- #
# list_tables
# --------------------------------------------------------------------------- #
class ListTablesArgs(BaseModel):
    """Không có tham số - nhưng vẫn cần một model để schema là hợp lệ."""

    model_config = {"extra": "forbid"}


def list_tables(args: ListTablesArgs, runtime: ToolRuntime) -> str:
    tables = _database(runtime).list_tables()
    if not tables:
        return "Database không có bảng nào."
    lines = ["Các bảng và view trong database:"]
    lines += [f"- {name} ({kind})" for name, kind in tables]
    lines.append("")
    lines.append("Dùng describe_table để xem cột của một bảng trước khi viết truy vấn.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# describe_table
# --------------------------------------------------------------------------- #
class DescribeTableArgs(BaseModel):
    model_config = {"extra": "forbid"}
    table: str = Field(description="tên bảng hoặc view, lấy từ list_tables")


def describe_table(args: DescribeTableArgs, runtime: ToolRuntime) -> str:
    info = _database(runtime).describe_table(args.table)
    lines = [f"Bảng {info['table']} - {info['row_count']} dòng", "", "Cột:"]
    for column in info["columns"]:
        flags = []
        if column["pk"]:
            flags.append("PK")
        if column["notnull"]:
            flags.append("NOT NULL")
        if column["default"] is not None:
            flags.append(f"default={column['default']}")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        lines.append(f"- {column['name']}: {column['type']}{suffix}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# run_query
# --------------------------------------------------------------------------- #
class RunQueryArgs(BaseModel):
    model_config = {"extra": "forbid"}
    sql: str = Field(
        min_length=1,
        description="MỘT câu SELECT hoặc WITH. Không được dùng INSERT/UPDATE/DELETE/PRAGMA/ATTACH.",
    )
    purpose: str = Field(
        min_length=3,
        description="một câu nói rõ truy vấn này nhằm trả lời điều gì",
    )


def run_query(args: RunQueryArgs, runtime: ToolRuntime) -> str:
    result = _database(runtime).run_select(args.sql)
    header = f"Truy vấn cho: {args.purpose}\nSQL đã chạy:\n{result.sql_executed}\n"
    return header + "\n" + result.to_markdown()


# --------------------------------------------------------------------------- #
# profile_column
# --------------------------------------------------------------------------- #
class ProfileColumnArgs(BaseModel):
    model_config = {"extra": "forbid"}
    table: str
    column: str
    top_values: int = Field(default=5, ge=1, le=20)


def profile_column(args: ProfileColumnArgs, runtime: ToolRuntime) -> str:
    """Tổng hợp một cột: số dòng, số NULL, số giá trị khác nhau, min/max, top giá trị.

    Có tool này vì nếu không, model sẽ phải tự viết 4-5 câu truy vấn để biết những
    điều rất cơ bản - tốn hạn mức và dễ sai. Một tool gộp sẵn các câu hỏi thường gặp
    làm agent đi nhanh hơn rõ rệt.

    Tên bảng/cột được kiểm với schema thật *trước khi* ghép vào SQL. Đây là chỗ duy
    nhất trong toolset ghép định danh vào câu truy vấn, nên nó cũng là chỗ duy nhất
    có thể bị SQL injection qua tên cột - và nó được chặn bằng allowlist lấy từ
    ``PRAGMA table_info``, không phải bằng escape.
    """
    db = _database(runtime)
    info = db.describe_table(args.table)
    names = [c["name"] for c in info["columns"]]
    if args.column not in names:
        raise ToolFailure(
            f"bảng {args.table!r} không có cột {args.column!r}",
            code="unknown_column",
            recoverable=True,
            hint=f"Các cột có sẵn: {', '.join(names)}",
        )

    table, column = args.table, args.column  # cả hai đã nằm trong allowlist của schema
    summary = db.run_select(
        f'SELECT COUNT(*) AS total, COUNT("{column}") AS non_null, '
        f'COUNT(DISTINCT "{column}") AS distinct_values, '
        f'MIN("{column}") AS min_value, MAX("{column}") AS max_value '
        f'FROM "{table}"'
    )
    top = db.run_select(
        f'SELECT "{column}" AS value, COUNT(*) AS n FROM "{table}" '
        f'GROUP BY "{column}" ORDER BY n DESC LIMIT {int(args.top_values)}'
    )

    total, non_null, distinct, min_value, max_value = summary.rows[0]
    lines = [
        f"Thống kê {args.table}.{args.column}:",
        f"- tổng số dòng: {total}",
        f"- có giá trị: {non_null} (NULL: {int(total) - int(non_null)})",
        f"- số giá trị khác nhau: {distinct}",
        f"- nhỏ nhất: {min_value!r}",
        f"- lớn nhất: {max_value!r}",
        "",
        f"{args.top_values} giá trị xuất hiện nhiều nhất:",
        top.to_markdown(),
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# save_finding - tool GHI, cần phê duyệt
# --------------------------------------------------------------------------- #
class SaveFindingArgs(BaseModel):
    model_config = {"extra": "forbid"}
    title: str = Field(min_length=3, max_length=200)
    summary: str = Field(min_length=10, max_length=4000)
    sql: str = Field(
        min_length=1,
        description="truy vấn đã tạo ra kết luận này, để người khác kiểm lại được",
    )


def save_finding(args: SaveFindingArgs, runtime: ToolRuntime) -> str:
    path = runtime.context.get("findings_path")
    if not path:  # pragma: no cover - lắp sai context
        raise ToolFailure(
            "harness chưa cấu hình nơi lưu finding",
            code="misconfigured",
            recoverable=False,
        )
    store = FindingStore(Path(path))
    record = store.append(
        {
            "title": args.title,
            "summary": args.summary,
            "sql": args.sql,
            "author": runtime.user_id,
            "saved_at": time.time(),
        }
    )
    return (
        f"Đã lưu finding #{record['id']} với tiêu đề {args.title!r}.\n"
        f"Tổng số finding hiện có: {record['total']}."
    )


class FindingStore:
    """Nơi lưu finding - một file JSON.

    Tách khỏi database SQLite có chủ ý: database của agent là **chỉ-đọc** và phải
    giữ nguyên như vậy. Một tool ghi không bao giờ được ghi vào cùng cái mà agent
    đang đọc - nếu không, agent có thể tự tạo ra "dữ liệu" rồi đọc lại chính nó như
    sự thật.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):  # pragma: no cover - file bị sửa tay
            return []
        return data if isinstance(data, list) else []

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        items = self.load()
        record = {**record, "id": len(items) + 1}
        items.append(record)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)  # ghi nguyên tử
        return {**record, "total": len(items)}

    def count(self) -> int:
        return len(self.load())


# --------------------------------------------------------------------------- #
# Đăng ký
# --------------------------------------------------------------------------- #
def build_sql_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ToolSpec(
                name="list_tables",
                description="Liệt kê các bảng và view trong database. Dùng trước tiên khi chưa biết schema.",
                args_model=ListTablesArgs,
                handler=list_tables,
                side_effect=SIDE_EFFECT_READ,
                required_permission=PERM_SQL_READ,
            ),
            ToolSpec(
                name="describe_table",
                description="Xem cột, kiểu dữ liệu và số dòng của một bảng.",
                args_model=DescribeTableArgs,
                handler=describe_table,
                side_effect=SIDE_EFFECT_READ,
                required_permission=PERM_SQL_READ,
            ),
            ToolSpec(
                name="run_query",
                description=(
                    "Chạy một câu SELECT trên database chỉ-đọc. Kết quả bị giới hạn số dòng "
                    "và có timeout. Phải nói rõ purpose."
                ),
                args_model=RunQueryArgs,
                handler=run_query,
                side_effect=SIDE_EFFECT_READ,
                required_permission=PERM_SQL_READ,
            ),
            ToolSpec(
                name="profile_column",
                description=(
                    "Thống kê nhanh một cột: số dòng, số NULL, số giá trị khác nhau, "
                    "min/max và các giá trị phổ biến nhất."
                ),
                args_model=ProfileColumnArgs,
                handler=profile_column,
                side_effect=SIDE_EFFECT_READ,
                required_permission=PERM_SQL_READ,
            ),
            ToolSpec(
                name="save_finding",
                description=(
                    "Lưu lại một kết luận phân tích để người khác xem. Hành động này GHI "
                    "dữ liệu ra ngoài và cần người phê duyệt."
                ),
                args_model=SaveFindingArgs,
                handler=save_finding,
                side_effect=SIDE_EFFECT_WRITE,
                required_permission=PERM_FINDINGS_WRITE,
                requires_approval=True,
                risk="Ghi một bản ghi mới vào sổ finding mà người khác sẽ đọc và tin.",
            ),
        ]
    )
