"""Lỗi của harness.

Phân biệt quan trọng nhất ở đây: **lỗi của model không phải lỗi của hệ thống.**

Model gọi tool không tồn tại, truyền sai tham số, viết SQL sai cú pháp - đó là
chuyện *bình thường* trong một harness. Model là một thành phần không đáng tin cậy
nằm trong vòng lặp, và việc của harness là biến mỗi sai sót đó thành một
``ToolMessage`` có nội dung hữu ích để model tự sửa ở lượt sau. Không có cái nào
trong số đó được phép làm sập episode.

Vì vậy chỉ có hai nhóm:

* ``ToolFailure`` - mọi thứ sai ở phía model/tool. Được *chuyển thành thông điệp*
  cho model, không raise ra ngoài vòng lặp. ``recoverable`` nói model có nên thử
  lại hay không.
* ``HarnessError`` - harness bị cấu hình sai hoặc bị vi phạm bất biến. Đây mới là
  lỗi thật: raise ra ngoài, episode dừng.
"""

from __future__ import annotations

from typing import Any


class HarnessError(Exception):
    """Harness bị cấu hình sai / bất biến bị vi phạm. Không phải lỗi của model."""

    code = "harness_error"


class ToolNotRegistered(HarnessError):
    """Dùng khi *code* hỏi một tool không có. Model hỏi sai tool thì là ToolFailure."""

    code = "tool_not_registered"


class DuplicateTool(HarnessError):
    code = "duplicate_tool"


class ModelNotConfigured(HarnessError):
    code = "model_not_configured"


class ToolFailure(Exception):
    """Một lần gọi tool thất bại - nhưng episode vẫn tiếp tục.

    ``recoverable=True``  -> model nên thử lại với tham số khác (sai cú pháp SQL,
                             sai tên bảng, thiếu tham số).
    ``recoverable=False`` -> thử lại y nguyên sẽ lại thất bại (bị policy chặn,
                             người dùng từ chối, vượt hạn mức).
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "tool_failed",
        recoverable: bool = True,
        hint: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.recoverable = recoverable
        # hint là thứ model đọc để sửa. "Bảng không tồn tại" ít hữu ích hơn
        # "bảng không tồn tại; các bảng có sẵn: orders, customers".
        self.hint = hint
        self.details = details or {}

    def to_tool_content(self) -> str:
        """Nội dung đưa vào ToolMessage cho model đọc."""
        parts = [f"LỖI [{self.code}]: {self.message}"]
        if self.hint:
            parts.append(f"GỢI Ý: {self.hint}")
        parts.append(
            "Có thể thử lại với tham số khác."
            if self.recoverable
            else "KHÔNG thử lại lời gọi này; hãy chọn cách khác hoặc kết luận với dữ liệu đang có."
        )
        return "\n".join(parts)


class PolicyDenied(ToolFailure):
    """Policy chặn trước khi tool chạy. Không bao giờ recoverable."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("code", "policy_denied")
        kwargs["recoverable"] = False
        super().__init__(message, **kwargs)


class ApprovalDenied(ToolFailure):
    """Người dùng từ chối phê duyệt."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("code", "approval_denied")
        kwargs["recoverable"] = False
        super().__init__(message, **kwargs)


class ArgumentInvalid(ToolFailure):
    """Tham số không khớp schema của tool."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("code", "invalid_arguments")
        kwargs.setdefault("recoverable", True)
        super().__init__(message, **kwargs)
