"""Test guard SQL và tầng truy cập database.

SQL ở đây do **model viết**, nên guard là một bề mặt bảo mật thật. Hai bug đã bị
bắt bởi bộ test này và được khoá lại bên dưới:

* truy vấn hợp lệ chứa ``--`` hoặc ``/*`` **trong chuỗi** bị phá vì phân tích được
  thực hiện trên bản đã bỏ comment rồi *bản đó* bị đem đi chạy,
* dấu mở/đóng block comment nằm trong chuỗi che được một câu ``ATTACH`` khỏi phép
  đếm câu lệnh.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from harness.errors import ToolFailure
from toolsets.sql.db import ReadOnlyDatabase, SqlLimits
from toolsets.sql.guard import (
    apply_limit,
    ensure_read_only,
    mask_string_literals,
    normalize_for_analysis,
    split_statements,
)


# --------------------------------------------------------------------------- #
# Allowlist
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "select * from customers limit 3",
        "WITH x AS (SELECT 1 AS a) SELECT a FROM x",
        "  SELECT 2  ",
        "SELECT 2;",
        "SELECT 2;;;",
        "/* ghi chú */ SELECT 3",
        "SELECT 4 -- ghi chú cuối dòng",
    ],
)
def test_read_statements_are_allowed(sql: str) -> None:
    assert ensure_read_only(sql)


@pytest.mark.parametrize(
    "sql,expected_code",
    [
        ("DELETE FROM customers", "not_read_only"),
        ("INSERT INTO customers VALUES (1)", "not_read_only"),
        ("UPDATE customers SET name='x'", "not_read_only"),
        ("DROP TABLE customers", "not_read_only"),
        ("CREATE TABLE t(a int)", "not_read_only"),
        # Hai câu dưới là lý do dùng allowlist thay vì denylist: trên một connection
        # mode=ro, SQLite VẪN cho phép chúng (đã đo).
        ("ATTACH DATABASE ':memory:' AS m", "not_read_only"),
        ("PRAGMA writable_schema = 1", "not_read_only"),
        ("VACUUM", "not_read_only"),
        ("", "empty_sql"),
        ("   ", "empty_sql"),
        ("SELECT 1; SELECT 2", "multiple_statements"),
        ("WITH x AS (SELECT 1) DELETE FROM customers", "forbidden_keyword"),
        ("SELECT load_extension('evil.so')", "forbidden_keyword"),
    ],
)
def test_dangerous_statements_are_blocked(sql: str, expected_code: str) -> None:
    with pytest.raises(ToolFailure) as exc:
        ensure_read_only(sql)
    assert exc.value.code == expected_code


def test_blocked_statement_tells_the_model_what_to_do() -> None:
    with pytest.raises(ToolFailure) as exc:
        ensure_read_only("DELETE FROM customers")
    content = exc.value.to_tool_content()
    assert "chỉ-đọc" in content
    assert "SELECT" in content


# --------------------------------------------------------------------------- #
# Hai bug đã bắt được
# --------------------------------------------------------------------------- #
def test_comment_marker_inside_string_does_not_break_a_valid_query() -> None:
    """BUG 1: ``SELECT '--;' AS x`` là hợp lệ và phải chạy đúng.

    Trước khi mask chuỗi, phân tích bỏ comment biến nó thành ``SELECT '`` và truy vấn
    hợp lệ của người dùng bị phá với một lỗi cú pháp vô nghĩa.
    """
    assert ensure_read_only("SELECT '--;' AS x") == "SELECT '--;' AS x"
    assert ensure_read_only("SELECT '/* x */' AS y") == "SELECT '/* x */' AS y"


def test_string_literal_cannot_hide_a_second_statement() -> None:
    """BUG 2: dấu mở/đóng block comment trong chuỗi che một câu ATTACH.

    Nếu bỏ comment *trước* khi mask chuỗi thì cả đoạn ATTACH biến mất khỏi bản phân
    tích và guard tưởng đây là một câu SELECT duy nhất.
    """
    attack = "SELECT '/*' AS a, 1; ATTACH DATABASE ':memory:' AS m; SELECT '*/' AS b"
    with pytest.raises(ToolFailure) as exc:
        ensure_read_only(attack)
    assert exc.value.code == "multiple_statements"


def test_keyword_inside_string_is_not_a_forbidden_keyword() -> None:
    """``WHERE notes LIKE '%delete%'`` là truy vấn đọc bình thường."""
    sql = "SELECT name FROM customers WHERE notes LIKE '%delete%' LIMIT 3"
    assert ensure_read_only(sql) == sql


def test_mask_string_literals_keeps_structure() -> None:
    assert mask_string_literals("SELECT 'abc' FROM t") == "SELECT '' FROM t"
    assert mask_string_literals('SELECT "col;x" FROM t') == 'SELECT "" FROM t'
    assert ";" not in mask_string_literals("SELECT ';;;' AS x")


def test_normalize_is_never_executed() -> None:
    """Bản phân tích mất nội dung chuỗi nên không được đem chạy; guard trả về SQL gốc."""
    sql = "SELECT 'quan trọng' AS x"
    assert "quan trọng" not in normalize_for_analysis(sql)
    assert "quan trọng" in ensure_read_only(sql)


def test_split_statements_respects_quotes() -> None:
    assert split_statements("SELECT ';' AS a") == ["SELECT ';' AS a"]
    assert len(split_statements("SELECT 1; SELECT 2")) == 2


# --------------------------------------------------------------------------- #
# LIMIT
# --------------------------------------------------------------------------- #
def test_limit_is_wrapped_not_appended() -> None:
    """Bọc thay vì nối: nối sẽ sai với ORDER BY lồng, UNION, hay comment ở cuối."""
    sql, wrapped = apply_limit("SELECT a FROM t ORDER BY a", 10)
    assert wrapped is True
    assert sql.startswith("SELECT * FROM (")
    assert sql.rstrip().endswith("LIMIT 10")


def test_existing_limit_is_respected() -> None:
    sql, wrapped = apply_limit("SELECT a FROM t LIMIT 5", 100)
    assert wrapped is False
    assert sql == "SELECT a FROM t LIMIT 5"


def test_limit_in_a_comment_does_not_count() -> None:
    _, wrapped = apply_limit("SELECT a FROM t -- limit 5", 10)
    assert wrapped is True


def test_wrapped_query_with_trailing_comment_is_still_valid_sql(tmp_path: Path) -> None:
    """Bọc phải dùng xuống dòng, nếu không ``)`` bị comment ăn mất.

    ``SELECT 1 -- x`` nối thêm `` ) LIMIT n`` trên cùng dòng sẽ thành lỗi cú pháp.
    """
    path = tmp_path / "t.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t(a INT)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.commit()
    conn.close()

    db = ReadOnlyDatabase(path, SqlLimits(max_rows=5))
    result = db.run_select("SELECT a FROM t -- lấy cột a")
    assert result.rows == [(1,)]


# --------------------------------------------------------------------------- #
# Tầng database
# --------------------------------------------------------------------------- #
def test_engine_is_read_only_even_if_guard_were_bypassed(demo_db: Path) -> None:
    """Lớp bảo vệ cứng: SQLite từ chối ghi bất kể guard có bỏ sót gì.

    Test này đi *vòng qua* guard và nói trực tiếp với engine - đúng để kiểm chứng
    rằng lớp 1 độc lập với lớp 2.
    """
    conn = sqlite3.connect(f"file:{demo_db.as_posix()}?mode=ro", uri=True)
    try:
        for statement in (
            "DELETE FROM customers",
            "INSERT INTO customers (name, signup_date) VALUES ('x','2026-01-01')",
            "CREATE TABLE evil(a INT)",
        ):
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                conn.execute(statement)
    finally:
        conn.close()


def test_row_limit_is_enforced_and_reported(demo_db: Path) -> None:
    db = ReadOnlyDatabase(demo_db, SqlLimits(max_rows=5))
    result = db.run_select("SELECT * FROM order_items")
    assert len(result.rows) == 5
    assert result.truncated_rows is True
    assert "KHÔNG phải toàn bộ dữ liệu" in result.to_markdown()


def test_small_result_is_not_reported_as_truncated(demo_db: Path) -> None:
    db = ReadOnlyDatabase(demo_db, SqlLimits(max_rows=100))
    result = db.run_select("SELECT * FROM customers LIMIT 3")
    assert result.truncated_rows is False
    assert "KHÔNG phải toàn bộ" not in result.to_markdown()


def test_long_cell_is_clipped(demo_db: Path) -> None:
    """Một ô TEXT dài có thể làm nổ context trước khi hạn mức kết quả kịp chặn."""
    db = ReadOnlyDatabase(demo_db, SqlLimits(max_rows=50, max_cell_chars=20))
    result = db.run_select("SELECT notes FROM customers WHERE notes IS NOT NULL")
    assert result.truncated_cells > 0
    assert any("…[cắt]" in str(cell) for row in result.rows for cell in row)


def test_unknown_table_lists_available_tables(demo_db: Path) -> None:
    db = ReadOnlyDatabase(demo_db, SqlLimits())
    with pytest.raises(ToolFailure) as exc:
        db.describe_table("khong_ton_tai")
    assert exc.value.code == "unknown_table"
    assert "customers" in (exc.value.hint or "")


def test_unknown_column_error_guides_the_model(demo_db: Path) -> None:
    db = ReadOnlyDatabase(demo_db, SqlLimits())
    with pytest.raises(ToolFailure) as exc:
        db.run_select("SELECT khong_co_cot FROM customers")
    assert exc.value.code == "unknown_object"
    assert "describe_table" in (exc.value.hint or "")


def test_query_timeout_is_enforced(demo_db: Path) -> None:
    """``SELECT`` hợp lệ nhưng nặng phải bị cắt, không treo episode.

    Join Descartes bốn bảng trên 400 dòng là đúng loại truy vấn model dễ viết ra.
    """
    db = ReadOnlyDatabase(demo_db, SqlLimits(max_rows=10, max_query_seconds=0.05))
    with pytest.raises(ToolFailure) as exc:
        db.run_select(
            "SELECT COUNT(*) FROM order_items a, order_items b, order_items c, order_items d"
        )
    assert exc.value.code == "query_timeout"
    assert exc.value.recoverable is True
    assert "WHERE" in (exc.value.hint or "")


def test_missing_database_is_a_clear_failure(tmp_path: Path) -> None:
    db = ReadOnlyDatabase(tmp_path / "khong-co.db", SqlLimits())
    with pytest.raises(ToolFailure) as exc:
        db.list_tables()
    assert exc.value.code == "database_missing"
    assert exc.value.recoverable is False
    assert "seed" in (exc.value.hint or "")


def test_view_is_listed_and_describable(demo_db: Path) -> None:
    db = ReadOnlyDatabase(demo_db, SqlLimits())
    names = dict(db.list_tables())
    assert names["order_revenue"] == "view"
    assert db.describe_table("order_revenue")["columns"]
