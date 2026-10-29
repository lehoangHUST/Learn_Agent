"""Test toolset SQL - phần domain.

Tập trung vào chỗ dễ sai nhất của một toolset do model điều khiển: nơi **tên do
model cung cấp được ghép vào SQL**. Trong toolset này chỉ có đúng một chỗ như vậy
(``profile_column``), và nó được bảo vệ bằng allowlist lấy từ schema thật, không
bằng escape chuỗi.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.errors import ToolFailure
from harness.registry import ToolRuntime
from toolsets.sql.db import ReadOnlyDatabase, SqlLimits
from toolsets.sql.tools import (
    DescribeTableArgs,
    FindingStore,
    ListTablesArgs,
    ProfileColumnArgs,
    RunQueryArgs,
    SaveFindingArgs,
    build_sql_registry,
    describe_table,
    list_tables,
    profile_column,
    run_query,
    save_finding,
)


@pytest.fixture
def runtime(demo_db: Path, findings_path: Path) -> ToolRuntime:
    return ToolRuntime(
        user_id="tester",
        context={
            "database": ReadOnlyDatabase(demo_db, SqlLimits(max_rows=20, max_cell_chars=100)),
            "findings_path": findings_path,
        },
    )


# --------------------------------------------------------------------------- #
def test_list_tables_guides_the_next_step(runtime: ToolRuntime) -> None:
    """Kết quả tool nên nói model làm gì tiếp - rẻ hơn một lượt model để nó tự mò."""
    text = list_tables(ListTablesArgs(), runtime)
    assert "customers" in text
    assert "order_revenue (view)" in text
    assert "describe_table" in text


def test_describe_table_shows_types_and_flags(runtime: ToolRuntime) -> None:
    text = describe_table(DescribeTableArgs(table="customers"), runtime)
    assert "customer_id" in text and "PK" in text
    assert "NOT NULL" in text
    assert "dòng" in text  # số dòng, để model biết quy mô trước khi SELECT *


def test_run_query_echoes_the_executed_sql(runtime: ToolRuntime) -> None:
    """Trả lại SQL đã chạy: người đọc phải kiểm lại được, kể cả phần harness tự thêm."""
    text = run_query(
        RunQueryArgs(sql="SELECT COUNT(*) AS n FROM customers", purpose="Đếm khách hàng"),
        runtime,
    )
    assert "Đếm khách hàng" in text
    assert "SELECT COUNT(*) AS n FROM customers" in text


def test_profile_column_reports_nulls(runtime: ToolRuntime) -> None:
    """``customers.segment`` cố ý có nhiều NULL trong dữ liệu demo."""
    text = profile_column(ProfileColumnArgs(table="customers", column="segment"), runtime)
    assert "NULL:" in text
    assert "giá trị khác nhau" in text
    assert "xuất hiện nhiều nhất" in text


def test_profile_column_rejects_unknown_column_with_options(runtime: ToolRuntime) -> None:
    with pytest.raises(ToolFailure) as exc:
        profile_column(ProfileColumnArgs(table="customers", column="khong_co"), runtime)
    assert exc.value.code == "unknown_column"
    assert "segment" in (exc.value.hint or "")


@pytest.mark.parametrize(
    "malicious_column",
    [
        'name" FROM customers; DROP TABLE customers; --',
        "name) UNION SELECT 1,2,3,4,5 --",
        'notes" || (SELECT name FROM sqlite_master) || "',
    ],
)
def test_profile_column_blocks_identifier_injection(
    runtime: ToolRuntime, malicious_column: str
) -> None:
    """Tên cột là chỗ duy nhất ghép định danh vào SQL -> chặn bằng allowlist của schema.

    Không escape, không quote rồi hy vọng: tên phải **có trong** danh sách cột thật
    mà ``PRAGMA table_info`` trả về, nếu không thì dừng.
    """
    with pytest.raises(ToolFailure) as exc:
        profile_column(
            ProfileColumnArgs(table="customers", column=malicious_column), runtime
        )
    assert exc.value.code == "unknown_column"


def test_profile_column_rejects_unknown_table(runtime: ToolRuntime) -> None:
    with pytest.raises(ToolFailure) as exc:
        profile_column(ProfileColumnArgs(table="x; DROP TABLE y", column="a"), runtime)
    assert exc.value.code == "unknown_table"


# --------------------------------------------------------------------------- #
# save_finding + FindingStore
# --------------------------------------------------------------------------- #
def test_save_finding_writes_and_counts(runtime: ToolRuntime, findings_path: Path) -> None:
    text = save_finding(
        SaveFindingArgs(
            title="Kết luận thử",
            summary="Một kết luận đủ dài để qua validate.",
            sql="SELECT 1",
        ),
        runtime,
    )
    assert "#1" in text
    stored = json.loads(findings_path.read_text(encoding="utf-8"))
    assert stored[0]["author"] == "tester"
    assert stored[0]["sql"] == "SELECT 1"


def test_finding_store_append_is_atomic(tmp_path: Path) -> None:
    """Ghi qua file tạm rồi replace: không để lại file JSON nửa vời nếu process chết."""
    store = FindingStore(tmp_path / "f.json")
    store.append({"title": "a"})
    store.append({"title": "b"})
    assert store.count() == 2
    assert [r["id"] for r in store.load()] == [1, 2]
    # Không còn file tạm.
    assert not (tmp_path / "f.json.tmp").exists()


def test_finding_store_tolerates_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "f.json"
    path.write_text("{ không phải json hợp lệ", encoding="utf-8")
    store = FindingStore(path)
    assert store.load() == []  # không raise
    store.append({"title": "mới"})
    assert store.count() == 1


def test_save_finding_requires_a_substantial_summary() -> None:
    """Schema chặn finding rỗng nghĩa: nó là thứ người khác sẽ đọc và tin."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SaveFindingArgs(title="x", summary="ngắn", sql="SELECT 1")


def test_run_query_requires_a_purpose() -> None:
    """Buộc model phát biểu ý định trước khi viết SQL - và transcript đọc được."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RunQueryArgs(sql="SELECT 1", purpose="")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_registry_labels_risk_correctly() -> None:
    registry = build_sql_registry()
    by_name = {spec.name: spec for spec in registry.all()}

    for name in ("list_tables", "describe_table", "run_query", "profile_column"):
        assert by_name[name].side_effect == "read"
        assert by_name[name].requires_approval is False
        assert by_name[name].required_permission == "sql:read"

    write = by_name["save_finding"]
    assert write.side_effect == "write"
    assert write.requires_approval is True
    assert write.required_permission == "findings:write"
    assert write.risk


def test_misconfigured_context_fails_clearly(findings_path: Path) -> None:
    """Thiếu database trong context là lỗi lắp đặt, phải nói rõ và không retry được."""
    runtime = ToolRuntime(user_id="t", context={"findings_path": findings_path})
    with pytest.raises(ToolFailure) as exc:
        list_tables(ListTablesArgs(), runtime)
    assert exc.value.code == "misconfigured"
    assert exc.value.recoverable is False
