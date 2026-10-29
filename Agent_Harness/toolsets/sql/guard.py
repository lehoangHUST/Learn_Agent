"""Chặn SQL nguy hiểm trước khi nó chạm vào database.

Trong harness này, **SQL do model viết**. Không có cách nào kiểm tra trước mọi câu
truy vấn mà model sẽ nghĩ ra, nên biện pháp phải là *cấu trúc*, không phải *đoán*.

Ba lớp, mỗi lớp chặn thứ hai lớp kia không chặn
-----------------------------------------------
**Lớp 1 - engine chỉ-đọc** (``db.py``, URI ``mode=ro``). Đây là bảo đảm cứng nhất:
``INSERT``/``UPDATE``/``DELETE``/``CREATE`` thất bại ngay trong SQLite, bất kể guard
có bỏ sót gì. Đã kiểm chứng bằng cách chạy thật.

**Lớp 2 - allowlist câu lệnh** (file này). Lớp 1 *không* đủ, và đây là chỗ đáng nói:
đo trên SQLite thật, một connection ``mode=ro`` **vẫn cho phép**::

    ATTACH DATABASE '/tmp/writable.db' AS m   -- rồi ghi vào m
    PRAGMA writable_schema = 1                -- sửa schema qua đường khác

Nên một denylist kiểu "cấm DELETE, cấm DROP" là thua ngay từ đầu: nó chỉ cấm những
gì người viết nghĩ ra được. Ở đây dùng **allowlist**: câu lệnh phải bắt đầu bằng
``SELECT`` hoặc ``WITH``, và chỉ được là **một** câu lệnh. Mọi thứ khác bị từ chối
mà không cần biết nó làm gì.

**Lớp 3 - giới hạn tài nguyên** (``db.py``): LIMIT bắt buộc và timeout cắt câu
truy vấn. Một ``SELECT`` hoàn toàn hợp lệ vẫn có thể là một join Descartes ba bảng
chạy mãi không xong - và model rất dễ viết ra đúng thứ đó.
"""

from __future__ import annotations

import re

from harness.errors import ToolFailure

# Chỉ hai dạng câu lệnh được chạy. Allowlist, không phải denylist.
_ALLOWED_PREFIXES = ("select", "with")

# Vẫn giữ một denylist token *bên cạnh* allowlist - không phải để bảo mật (allowlist
# đã lo), mà để thông điệp lỗi nói rõ model đã làm gì sai. "Không được dùng ATTACH"
# hữu ích hơn "câu lệnh phải bắt đầu bằng SELECT".
_FORBIDDEN_TOKENS = (
    "attach",
    "detach",
    "pragma",
    "vacuum",
    "insert",
    "update",
    "delete",
    "replace",
    "drop",
    "alter",
    "create",
    "reindex",
    "analyze",
    "begin",
    "commit",
    "rollback",
    "savepoint",
    "release",
    "load_extension",
)

_LIMIT_RE = re.compile(r"\blimit\b", re.IGNORECASE)
_COMMENT_BLOCK_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_COMMENT_LINE_RE = re.compile(r"--[^\n]*")


def mask_string_literals(sql: str) -> str:
    """Thay nội dung bên trong chuỗi/định danh bằng rỗng, giữ nguyên độ dài cấu trúc.

    Phải làm bước này **trước** khi bỏ comment, và đây là chỗ hai bug thật đã bị bắt
    khi thử trên SQLite:

    1. ``SELECT '--;' AS x`` là truy vấn hợp lệ. Nếu bỏ comment trước, chuỗi ``--;'``
       bị coi là comment và câu lệnh bị cắt thành ``SELECT '`` - truy vấn hợp lệ của
       người dùng bị **phá**, báo lỗi cú pháp vô nghĩa.
    2. ``SELECT '/*' AS a, 1; ATTACH DATABASE ... ; SELECT '*/' AS b`` - dấu mở/đóng
       block comment nằm trong chuỗi. Nếu bỏ comment trước, cả đoạn ``ATTACH`` biến
       mất khỏi bản phân tích và guard tưởng đây là *một* câu SELECT.

    Mask trước thì cả hai biến mất: nội dung chuỗi không còn ảnh hưởng tới việc
    phân tích, và dấu ``;`` hay ``/*`` bên trong chuỗi không còn nghĩa cú pháp.
    """
    out: list[str] = []
    quote: str | None = None
    for char in sql:
        if quote:
            if char == quote:
                quote = None
                out.append(char)
            # nội dung bên trong chuỗi bị bỏ đi hoàn toàn
            continue
        if char in ("'", '"'):
            quote = char
            out.append(char)
            continue
        out.append(char)
    return "".join(out)


def strip_sql_comments(sql: str) -> str:
    """Bỏ comment. Luôn gọi sau ``mask_string_literals`` khi dùng để phân tích."""
    return _COMMENT_LINE_RE.sub(" ", _COMMENT_BLOCK_RE.sub(" ", sql))


def normalize_for_analysis(sql: str) -> str:
    """Bản dùng để *phân tích*: mask chuỗi rồi bỏ comment.

    Bản này KHÔNG bao giờ được đem đi chạy - nó đã mất nội dung chuỗi. Câu được chạy
    luôn là SQL gốc của model.
    """
    return strip_sql_comments(mask_string_literals(sql))


def split_statements(sql: str) -> list[str]:
    """Tách câu lệnh theo ``;``, bỏ qua ``;`` nằm trong chuỗi.

    Viết tay thay vì ``sql.split(";")`` vì ``SELECT ';' AS x`` là hợp lệ và không
    phải hai câu lệnh. SQLite chỉ dùng dấu nháy đơn cho chuỗi; dấu nháy kép là
    định danh, nên cả hai đều phải được bỏ qua.
    """
    statements: list[str] = []
    current: list[str] = []
    quote: str | None = None
    for char in sql:
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            current.append(char)
            continue
        if char == ";":
            statements.append("".join(current))
            current = []
            continue
        current.append(char)
    statements.append("".join(current))
    return [s.strip() for s in statements if s.strip()]


def ensure_read_only(sql: str) -> str:
    """Kiểm tra và trả về câu lệnh duy nhất đã chuẩn hoá. Sai -> ``ToolFailure``.

    ``ToolFailure`` chứ không phải exception của harness: model viết SQL sai là
    chuyện thường, và nó cần đọc được lý do để sửa.
    """
    if not sql or not sql.strip():
        raise ToolFailure(
            "câu truy vấn rỗng",
            code="empty_sql",
            hint="Hãy gửi một câu SELECT.",
        )

    analysed = normalize_for_analysis(sql)
    statements = split_statements(analysed)

    if len(statements) > 1:
        raise ToolFailure(
            f"chỉ chạy được một câu lệnh mỗi lần, nhận được {len(statements)}",
            code="multiple_statements",
            hint="Hãy gửi từng câu SELECT một.",
            details={"statement_count": len(statements)},
        )
    if not statements:
        raise ToolFailure("không tìm thấy câu lệnh nào", code="empty_sql")

    lowered = statements[0].lower()
    first_word = re.split(r"\s+|\(", lowered, maxsplit=1)[0]

    if first_word not in _ALLOWED_PREFIXES:
        raise ToolFailure(
            f"chỉ cho phép câu lệnh đọc (SELECT / WITH), nhận được {first_word.upper()!r}",
            code="not_read_only",
            recoverable=True,
            hint="Database này là chỉ-đọc. Hãy viết một câu SELECT.",
            details={"first_word": first_word},
        )

    # Token bị cấm ở bất kỳ vị trí nào - bắt được cả `WITH x AS (...) DELETE ...`.
    for token in _FORBIDDEN_TOKENS:
        if re.search(rf"\b{token}\b", lowered):
            raise ToolFailure(
                f"từ khoá {token.upper()!r} không được phép trong truy vấn",
                code="forbidden_keyword",
                recoverable=True,
                hint="Chỉ dùng SELECT, WITH, JOIN, WHERE, GROUP BY, ORDER BY, LIMIT.",
                details={"keyword": token},
            )

    # Trả về SQL **gốc** (chỉ bỏ dấu ; ở cuối), không trả về bản đã phân tích: bản
    # đó đã bị mask nội dung chuỗi nên chạy nó sẽ cho kết quả sai hoặc lỗi cú pháp.
    return sql.strip().rstrip(";").strip()


def apply_limit(statement: str, max_rows: int) -> tuple[str, bool]:
    """Bảo đảm câu truy vấn không trả về quá ``max_rows`` dòng.

    Trả về ``(sql, đã_bọc)``. Nếu câu lệnh chưa có ``LIMIT``, bọc nó lại thay vì
    nối ``LIMIT`` vào cuối: nối chuỗi sẽ sai với ``... ORDER BY x LIMIT 5`` lồng
    trong subquery, hoặc với ``UNION``. Bọc thành ``SELECT * FROM (<sql>) LIMIT n``
    luôn đúng về cú pháp.

    ``LIMIT`` do model tự viết được giữ nguyên: nếu model xin 10 dòng thì không có
    lý do trả về 200. Nhưng ``max_rows`` vẫn là trần cứng ở tầng đọc cursor
    (``db.run_select``), nên một ``LIMIT 100000`` vẫn bị cắt.
    """
    if _LIMIT_RE.search(normalize_for_analysis(statement)):
        return statement, False
    return f"SELECT * FROM (\n{statement}\n) LIMIT {int(max_rows)}", True
