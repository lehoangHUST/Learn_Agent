"""State của một episode.

Một **episode** = một lần harness chạy để giải quyết một mục tiêu của người dùng.
Nó có thể gồm nhiều lượt gọi model và nhiều lượt gọi tool, và có thể bị **tạm dừng
giữa đường** để chờ người phê duyệt rồi chạy tiếp (có thể là nhiều giờ sau).

Hệ quả của việc "tạm dừng rồi chạy tiếp": mọi thứ cần để tiếp tục phải **serialize
được**, vì nó đi qua checkpoint. Không có object sống, không có connection, không có
closure nào trong state. Model client, DB connection và tool registry nằm ở
``HarnessContext`` (``context_schema``) - thứ được dựng lại ở mỗi lần invoke.

Về reducer
----------
* ``messages`` -> ``add_messages``: đây là bộ nhớ làm việc của model. Phải tích luỹ,
  và phải upsert theo id để sửa được một message đã có.
* ``steps`` -> nối: transcript của episode, chỉ ghi thêm.
* Các bộ đếm (``model_calls``, ``tool_calls_total``, ...) -> **ghi đè**. Node đọc giá
  trị hiện tại rồi trả về giá trị mới. An toàn vì graph này tuần tự: trong một
  super-step chỉ có đúng một node ghi mỗi bộ đếm. Nếu sau này cho phép chạy tool
  song song trong cùng một super-step thì các field này **phải** đổi sang reducer
  cộng dồn, nếu không sẽ mất đếm (hoặc ``InvalidUpdateError``). Ghi chú này là
  có chủ ý: nó là điều kiện để thay đổi thiết kế, không phải một lời cảnh báo chung.
"""

from __future__ import annotations

import operator
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, NotRequired

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from harness.budget import BudgetPolicy
    from harness.model import ModelClient
    from harness.registry import ToolRegistry

# Trạng thái cuối của một episode.
STATUS_RUNNING = "running"
STATUS_ANSWERED = "answered"  # model đã trả lời xong
STATUS_STOPPED = "stopped"  # harness dừng vì hạn mức/policy, KHÔNG phải lỗi
STATUS_FAILED = "failed"  # lỗi thật của harness

# Lý do dừng - mã ổn định để log và test.
STOP_ANSWERED = "model_finished"
STOP_MAX_MODEL_CALLS = "budget_max_model_calls"
STOP_MAX_TOOL_CALLS = "budget_max_tool_calls"
STOP_DEADLINE = "budget_deadline"
STOP_CONTEXT_BUDGET = "budget_result_chars"
STOP_NO_PROGRESS = "no_progress"


class StepRecord(TypedDict):
    """Một bước trong transcript. Chỉ chứa dữ liệu, không chứa object."""

    index: int
    kind: str  # "model" | "tool" | "approval" | "stop"
    name: str
    status: str  # "ok" | "error" | "denied" | "skipped"
    duration_ms: float
    detail: NotRequired[dict[str, Any]]


class PendingApproval(TypedDict):
    """Lời gọi tool đang chờ người phê duyệt.

    Nằm trong state (không nằm trong context) vì nó phải sống qua checkpoint: người
    phê duyệt có thể quay lại sau nhiều giờ, ở một process khác.
    """

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    reason: str
    risk: str


class HarnessState(TypedDict):
    # --- định danh ---
    episode_id: str
    thread_id: str
    user_id: str

    # --- mục tiêu ---
    goal: str

    # --- bộ nhớ làm việc của model ---
    messages: Annotated[list[AnyMessage], add_messages]

    # --- sổ hạn mức (ghi đè; xem docstring module) ---
    model_calls: int
    tool_calls_total: int
    tool_calls_by_name: dict[str, int]
    result_chars_total: int
    # Thời gian *làm việc thật*, KHÔNG tính thời gian chờ người phê duyệt.
    # Dùng deadline tuyệt đối tính từ lúc bắt đầu episode sẽ làm một lần chờ phê
    # duyệt 1 tiếng giết chết episode ngay khi nó được tiếp tục.
    elapsed_active_seconds: float

    # --- tín hiệu điều khiển ---
    # Các lời gọi tool mà lượt model vừa rồi xin. Đây là thứ router đọc để quyết
    # định đi sang execute_tools hay kết thúc.
    #
    # Vì sao KHÔNG suy từ "message cuối cùng có tool_calls không": ``add_messages``
    # upsert theo ``id``. Nếu model trả về một message có id đã tồn tại, message đó
    # bị **ghi đè tại chỗ** và giữ nguyên vị trí cũ trong danh sách - nên message
    # cuối cùng lại là ToolMessage của lượt trước, và router kết luận sai rằng model
    # đã trả lời xong. Đúng lỗi này đã làm kịch bản "model lặp vô hạn" lặng lẽ kết
    # thúc với status "answered" trước khi được phát hiện.
    #
    # Tín hiệu điều khiển tường minh thì không phụ thuộc vào thứ tự hay danh tính
    # của message.
    pending_tool_calls: list[dict[str, Any]]

    # --- kết quả ---
    status: str
    stop_reason: str | None
    final_answer: str

    # --- transcript ---
    steps: Annotated[list[StepRecord], operator.add]

    # --- human-in-the-loop ---
    pending_approval: PendingApproval | None


class HarnessInput(TypedDict):
    """Hợp đồng đầu vào. Người gọi không set được bộ đếm hay trạng thái."""

    goal: str
    episode_id: str
    thread_id: str
    user_id: str


class HarnessOutput(TypedDict):
    episode_id: str
    thread_id: str
    status: str
    stop_reason: str | None
    final_answer: str
    model_calls: int
    tool_calls_total: int
    tool_calls_by_name: dict[str, int]
    elapsed_active_seconds: float
    steps: list[StepRecord]
    pending_approval: PendingApproval | None


@dataclass(slots=True)
class HarnessContext:
    """Dependency chỉ đọc của một lần invoke - KHÔNG đi vào checkpoint.

    Ba thứ này không serialize được và cũng không nên: model client giữ HTTP
    connection, registry giữ hàm Python, policy là cấu hình của process.
    Chúng được dựng lại ở mỗi lần invoke, kể cả lần invoke để resume sau phê duyệt.
    """

    model: ModelClient
    registry: ToolRegistry
    budget: BudgetPolicy
    user_id: str
    # Quyền của principal. Policy so sánh với quyền mà từng tool đòi.
    permissions: frozenset[str] = field(default_factory=frozenset)
    # Tham số nghiệp vụ truyền cho tool handler (ví dụ đường dẫn DB).
    tool_context: dict[str, Any] = field(default_factory=dict)


def new_episode_id() -> str:
    return "ep_" + uuid.uuid4().hex[:16]


def new_thread_id() -> str:
    return "th_" + uuid.uuid4().hex[:16]


def initial_state(
    *,
    goal: str,
    user_id: str,
    episode_id: str | None = None,
    thread_id: str | None = None,
) -> HarnessInput:
    return {
        "goal": goal,
        "episode_id": episode_id or new_episode_id(),
        "thread_id": thread_id or new_thread_id(),
        "user_id": user_id,
    }


def fresh_counters() -> dict[str, Any]:
    """Giá trị khởi tạo cho mọi field không thuộc ``HarnessInput``.

    Node ``prepare`` trả về dict này. Không gửi được từ input: ``input_schema`` hẹp
    nên LangGraph lọc bỏ im lặng mọi key ngoài schema đó.
    """
    return {
        "model_calls": 0,
        "tool_calls_total": 0,
        "tool_calls_by_name": {},
        "result_chars_total": 0,
        "elapsed_active_seconds": 0.0,
        "pending_tool_calls": [],
        "status": STATUS_RUNNING,
        "stop_reason": None,
        "final_answer": "",
        "pending_approval": None,
    }


def now() -> float:
    return time.perf_counter()
