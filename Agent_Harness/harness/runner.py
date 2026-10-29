"""Chạy episode, kể cả phần tạm dừng chờ phê duyệt.

Đây là nơi thể hiện rõ nhất vì sao harness cần checkpointer: một episode có thể
**dừng giữa đường**, trả về cho người gọi một danh sách hành động cần phê duyệt, rồi
được tiếp tục bởi một request khác, ở một process khác, nhiều giờ sau.

Hai lần invoke, một episode:

* ``start()``  -> nếu gặp tool cần phê duyệt, LangGraph dừng và ``invoke`` trả về
  thêm khoá ``__interrupt__``. Trạng thái lúc này là ``awaiting_approval``.
* ``resume()`` -> gọi lại với ``Command(resume=<quyết định>)``. Node ``execute_tools``
  **chạy lại từ đầu**, nhưng pha lập kế hoạch của nó thuần nên không có gì xảy ra
  hai lần; lần này ``interrupt()`` trả về quyết định thay vì dừng.

``context_factory`` được gọi ở **mỗi** lần invoke, kể cả lần resume. Đó là điểm của
``context_schema``: model client và DB connection không serialize được, nên chúng
không nằm trong checkpoint và phải được dựng lại. Nếu lỡ nhét chúng vào state thì
resume sẽ thất bại ở bước ghi checkpoint - và thất bại đó chỉ xuất hiện khi có một
lần phê duyệt thật, tức là muộn.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langgraph.types import Command

from harness.state import (
    HarnessContext,
    initial_state,
    new_episode_id,
    new_thread_id,
)

logger = logging.getLogger(__name__)

STATUS_AWAITING_APPROVAL = "awaiting_approval"


@dataclass(slots=True)
class EpisodeResult:
    status: str
    episode_id: str
    thread_id: str
    output: dict[str, Any] = field(default_factory=dict)
    # Danh sách lời gọi tool đang chờ phê duyệt (rỗng nếu không chờ).
    pending: list[dict[str, Any]] = field(default_factory=list)

    @property
    def awaiting_approval(self) -> bool:
        return self.status == STATUS_AWAITING_APPROVAL

    @property
    def final_answer(self) -> str:
        return str(self.output.get("final_answer") or "")


def _coerce_resume(decisions: Any) -> Any:
    """Chuẩn hoá giá trị resume trước khi đưa vào LangGraph.

    Hai lý do, cả hai đều phát hiện bằng test:

    1. **``None`` làm LangGraph 1.0.1 crash.** ``Command(resume=None)`` đi vào nhánh
       ``UnboundLocalError: resume_is_map`` trong ``langgraph/pregel/_loop.py``. Đó
       là bug của thư viện, nhưng hệ quả thuộc về ta: một client gửi body thiếu
       trường sẽ làm lỗi 500 thay vì một lần từ chối gọn gàng.
    2. **``{}`` khiến episode treo im lặng** - LangGraph coi giá trị falsy như "chưa
       có resume", nên graph dừng lại ở interrupt y như trước và client không hiểu
       vì sao không có gì xảy ra.

    Cả hai được quy về cùng một thứ: **từ chối tường minh**. Không trả lời thì không
    phải là đồng ý, và trạng thái của episode phải xác định.
    """
    if decisions is None or (isinstance(decisions, dict) and not decisions):
        return {
            "approved": False,
            "reason": "không nhận được quyết định phê duyệt nào (mặc định: từ chối)",
        }
    return decisions


class HarnessRunner:
    def __init__(
        self,
        graph: Any,
        context_factory: Callable[[str], HarnessContext],
        *,
        recursion_limit: int = 60,
    ) -> None:
        self._graph = graph
        self._context_factory = context_factory
        self._recursion_limit = recursion_limit

    def start(
        self,
        goal: str,
        *,
        user_id: str,
        thread_id: str | None = None,
        episode_id: str | None = None,
    ) -> EpisodeResult:
        thread_id = thread_id or new_thread_id()
        episode_id = episode_id or new_episode_id()
        payload = initial_state(
            goal=goal, user_id=user_id, episode_id=episode_id, thread_id=thread_id
        )
        logger.info(
            "episode_start",
            extra={"episode_id": episode_id, "thread_id": thread_id, "user_id": user_id},
        )
        return self._invoke(payload, user_id=user_id, thread_id=thread_id, episode_id=episode_id)

    def resume(
        self,
        thread_id: str,
        decisions: Any,
        *,
        user_id: str,
    ) -> EpisodeResult:
        """Tiếp tục một episode đang chờ phê duyệt.

        ``decisions`` nhận nhiều dạng (bool, dict chung, dict theo tool_call_id) -
        xem ``_normalize_decisions`` trong ``loop.py``. Giá trị không hiểu được bị
        coi là **từ chối**: cổng phê duyệt không được phép mở vì một payload méo mó.
        """
        logger.info("episode_resume", extra={"thread_id": thread_id, "user_id": user_id})
        return self._invoke(
            Command(resume=_coerce_resume(decisions)),
            user_id=user_id,
            thread_id=thread_id,
            episode_id="",
        )

    def get_state(self, thread_id: str) -> Any:
        return self._graph.get_state({"configurable": {"thread_id": thread_id}})

    def context_for(self, user_id: str) -> HarnessContext:
        """Context mà một người dùng sẽ nhận - để tra tool/quyền mà không chạy episode.

        Phơi ra như API công khai thay vì để tầng trên chạm vào ``_context_factory``:
        endpoint ``/tools`` cần biết agent *có thể* làm gì, và đó là thông tin vận
        hành hợp lệ, không phải chi tiết nội bộ.
        """
        return self._context_factory(user_id)

    # ------------------------------------------------------------------ #
    def _invoke(
        self,
        payload: Any,
        *,
        user_id: str,
        thread_id: str,
        episode_id: str,
    ) -> EpisodeResult:
        context = self._context_factory(user_id)
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self._recursion_limit,
            "run_name": "agent_harness",
            "metadata": {"thread_id": thread_id, "user_id": user_id},
        }
        output = self._graph.invoke(payload, config, context=context)

        interrupts = output.get("__interrupt__") or []
        if interrupts:
            value = getattr(interrupts[0], "value", {}) or {}
            requests = list(value.get("requests") or [])
            logger.info(
                "episode_awaiting_approval",
                extra={
                    "thread_id": thread_id,
                    "pending_tools": [r.get("tool_name") for r in requests],
                },
            )
            return EpisodeResult(
                status=STATUS_AWAITING_APPROVAL,
                episode_id=str(value.get("episode_id") or episode_id),
                thread_id=thread_id,
                output={k: v for k, v in output.items() if k != "__interrupt__"},
                pending=requests,
            )

        return EpisodeResult(
            status=str(output.get("status") or "unknown"),
            episode_id=str(output.get("episode_id") or episode_id),
            thread_id=str(output.get("thread_id") or thread_id),
            output=output,
        )
