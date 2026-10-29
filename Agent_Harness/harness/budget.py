"""Hạn mức - thứ làm nên khác biệt giữa một harness và một vòng lặp vô hạn.

Một harness đặt LLM vào vị trí điều khiển. Điều đó có nghĩa là **không ai biết
trước episode sẽ dài bao nhiêu**: model quyết định gọi tool bao nhiêu lần, và nó có
thể quyết định sai - lặp lại mãi một lời gọi, hoặc kéo một bảng 10 triệu dòng vào
context. Đây không phải giả thuyết; nó là chế độ thất bại *thường gặp nhất* của
agent tool-calling.

Nên harness phải có tám loại chặn, mỗi loại chặn một kiểu thất bại khác nhau:

===========================  ==================================================
Hạn mức                      Chặn kiểu thất bại nào
===========================  ==================================================
``max_model_calls``          vòng lặp model <-> tool không bao giờ kết thúc
``max_tool_calls``           model gọi tool quá nhiều dù mỗi lượt đều hợp lý
``max_tool_calls_per_tool``  model kẹt ở MỘT tool (gọi list_tables 20 lần)
``max_identical_tool_calls`` model lặp y nguyên một lời gọi (không tiến triển)
``max_active_seconds``       tool chậm/treo làm episode kéo dài vô hạn
``max_tool_result_chars``    một kết quả khổng lồ làm nổ context
``max_total_result_chars``   nhiều kết quả vừa phải cộng lại làm nổ context
``max_parallel_tool_calls``  một lượt assistant xin 30 tool, tiêu hết hạn mức
===========================  ==================================================

Hai nguyên tắc trong cách thực thi:

1. **Hết hạn mức là trạng thái kết thúc bình thường, không phải exception.**
   Episode dừng với ``status="stopped"`` và một ``stop_reason`` có mã ổn định, kèm
   theo những gì đã thu được. Người gọi vẫn nhận một response dùng được.
2. **Hạn mức được kiểm TRƯỚC khi tiêu tài nguyên**, không phải sau. Kiểm sau nghĩa
   là đã trả tiền cho lượt gọi model thứ 9 rồi mới nói "giới hạn là 8".
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from harness.state import (
    STOP_CONTEXT_BUDGET,
    STOP_DEADLINE,
    STOP_MAX_MODEL_CALLS,
    STOP_MAX_TOOL_CALLS,
    HarnessState,
)

TRUNCATION_NOTICE = (
    "\n\n[... kết quả bị cắt bởi harness: {dropped} ký tự nữa không được đưa vào. "
    "Hãy truy vấn hẹp hơn (thêm WHERE, chọn ít cột, dùng COUNT/GROUP BY) nếu cần "
    "phần còn lại.]"
)


@dataclass(frozen=True, slots=True)
class BudgetVerdict:
    allowed: bool
    stop_reason: str | None = None
    message: str = ""


ALLOWED = BudgetVerdict(allowed=True)


@dataclass(slots=True)
class BudgetPolicy:
    """Cấu hình hạn mức cho một episode.

    Số mặc định được chọn cho một agent phân tích dữ liệu: đủ để xem schema, thử
    vài query, sửa một lỗi, rồi kết luận. Chúng là **ngưỡng an toàn, không phải
    ngưỡng hiệu năng** - một episode lành mạnh thường dùng 3-5 lượt model.
    """

    max_model_calls: int = 8
    max_tool_calls: int = 24
    max_tool_calls_per_tool: dict[str, int] = field(default_factory=dict)
    # Lời gọi thứ N+1 y hệt nhau (cùng tool, cùng tham số) bị từ chối.
    max_identical_tool_calls: int = 3
    # Chỉ tính thời gian làm việc thật; thời gian chờ người phê duyệt không tính.
    max_active_seconds: float = 120.0
    max_tool_result_chars: int = 4000
    max_total_result_chars: int = 40_000
    # Một lượt assistant có thể xin nhiều tool cùng lúc; giới hạn để một lượt
    # không tiêu hết hạn mức.
    max_parallel_tool_calls: int = 4

    # ------------------------------------------------------------------ #
    # Kiểm trước khi tiêu tài nguyên
    # ------------------------------------------------------------------ #
    def check_before_model(self, state: HarnessState) -> BudgetVerdict:
        if state.get("model_calls", 0) >= self.max_model_calls:
            return BudgetVerdict(
                False,
                STOP_MAX_MODEL_CALLS,
                f"đã dùng hết {self.max_model_calls} lượt gọi model cho episode này",
            )
        return self._check_shared(state)

    def check_before_tool(self, state: HarnessState, tool_name: str) -> BudgetVerdict:
        if state.get("tool_calls_total", 0) >= self.max_tool_calls:
            return BudgetVerdict(
                False,
                STOP_MAX_TOOL_CALLS,
                f"đã dùng hết {self.max_tool_calls} lượt gọi tool cho episode này",
            )
        return self._check_shared(state)

    def _check_shared(self, state: HarnessState) -> BudgetVerdict:
        if state.get("elapsed_active_seconds", 0.0) >= self.max_active_seconds:
            return BudgetVerdict(
                False,
                STOP_DEADLINE,
                f"episode đã dùng {state.get('elapsed_active_seconds', 0):.1f}s thời gian "
                f"làm việc, vượt hạn mức {self.max_active_seconds}s",
            )
        if state.get("result_chars_total", 0) >= self.max_total_result_chars:
            return BudgetVerdict(
                False,
                STOP_CONTEXT_BUDGET,
                f"kết quả tool đã chiếm {state.get('result_chars_total', 0)} ký tự, "
                f"vượt hạn mức {self.max_total_result_chars}",
            )
        return ALLOWED

    # ------------------------------------------------------------------ #
    # Hạn mức ở mức từng tool - KHÔNG dừng episode, chỉ từ chối lời gọi
    # ------------------------------------------------------------------ #
    def per_tool_limit(self, tool_name: str) -> int | None:
        return self.max_tool_calls_per_tool.get(tool_name)

    def tool_quota_exceeded(self, state: HarnessState, tool_name: str) -> str | None:
        """Trả về thông điệp nếu tool này đã hết quota riêng, ngược lại None.

        Khác ``check_before_tool``: hết quota của *một* tool không làm episode dừng -
        model vẫn có thể dùng tool khác hoặc kết luận. Nên nó được trả về cho model
        dưới dạng lỗi tool, không phải lý do dừng.
        """
        limit = self.per_tool_limit(tool_name)
        if limit is None:
            return None
        used = (state.get("tool_calls_by_name") or {}).get(tool_name, 0)
        if used >= limit:
            return (
                f"tool {tool_name!r} đã được gọi {used} lần, đạt hạn mức {limit} lần "
                "cho mỗi episode"
            )
        return None

    # ------------------------------------------------------------------ #
    # Cắt kết quả
    # ------------------------------------------------------------------ #
    def truncate(self, text: str) -> tuple[str, int]:
        """Cắt một kết quả tool về trong hạn mức.

        Trả về ``(text_đã_cắt, số_ký_tự_bị_bỏ)``. Phần bị cắt được **nói ra** trong
        chính nội dung: model phải biết là nó đang xem dữ liệu không đầy đủ, nếu
        không nó sẽ kết luận "bảng có 50 dòng" trong khi 50 chỉ là phần nhìn thấy.
        """
        if len(text) <= self.max_tool_result_chars:
            return text, 0
        dropped = len(text) - self.max_tool_result_chars
        return text[: self.max_tool_result_chars] + TRUNCATION_NOTICE.format(dropped=dropped), dropped


def tool_call_signature(tool_name: str, arguments: dict[str, Any]) -> str:
    """Chữ ký ổn định của một lời gọi tool, để phát hiện lặp.

    ``sort_keys`` để thứ tự key trong dict do model sinh ra không làm hai lời gọi
    giống nhau trông khác nhau.
    """
    blob = json.dumps(
        {"tool": tool_name, "args": arguments}, sort_keys=True, ensure_ascii=False, default=str
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def count_identical_calls(state: HarnessState, signature: str) -> int:
    """Số lần lời gọi có chữ ký này đã chạy trong episode."""
    count = 0
    for step in state.get("steps") or []:
        if step.get("kind") == "tool" and (step.get("detail") or {}).get("signature") == signature:
            count += 1
    return count
