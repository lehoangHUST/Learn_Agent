"""Trừu tượng model + một model theo kịch bản để test.

Vì sao cần ``ScriptedModelClient``
----------------------------------
Trong một harness, **thứ cần test là harness, không phải model.** Những hành vi
quan trọng nhất phải kiểm lại đúng là những hành vi mà một model thật không chịu
tạo ra theo yêu cầu:

* model gọi một tool không tồn tại,
* model truyền tham số sai kiểu,
* model lặp y nguyên một lời gọi mãi không dừng,
* model xin một tool ghi dữ liệu,
* model xin 30 tool trong một lượt,
* model đọc một kết quả tool có chứa câu "bỏ qua quy tắc trước đó" rồi làm theo.

Không thể viết test tin cậy cho những tình huống đó bằng một model thật: nó không
tất định, tốn tiền, và phần lớn thời gian nó *không* sai theo cách ta cần. Model
theo kịch bản cho phép dựng chính xác từng tình huống và khẳng định harness xử lý
đúng. Đây là test double đúng chỗ, không phải chỗ lách.

Model thật (``OpenAIModelClient``) đi qua **cùng một interface**, nên đường chạy
production và đường test là một.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from langchain_core.messages import AIMessage, AnyMessage

from harness.errors import HarnessError, ModelNotConfigured

logger = logging.getLogger(__name__)


@runtime_checkable
class ModelClient(Protocol):
    """Hợp đồng duy nhất mà harness cần từ một model."""

    name: str

    def invoke(
        self,
        messages: Sequence[AnyMessage],
        *,
        tools: list[dict[str, Any]],
        config: dict[str, Any] | None = None,
    ) -> AIMessage:
        """Trả về một ``AIMessage``, có thể kèm ``tool_calls``."""
        ...


# --------------------------------------------------------------------------- #
# Model thật
# --------------------------------------------------------------------------- #
class OpenAIModelClient:
    """Bọc ``ChatOpenAI`` với tool-calling.

    Harness truyền ``tools`` ở **mỗi lượt** thay vì bind một lần: tập tool có thể
    đổi giữa các lượt (ví dụ thu hẹp sau khi đã có dữ liệu), và việc truyền tường
    minh làm cho "model được thấy tool nào" là một dữ kiện kiểm tra được trong test,
    không phải trạng thái ẩn trong một object đã bind.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1500,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ModelNotConfigured(
                "thiếu OPENAI_API_KEY. Đặt key, hoặc chạy với --scripted để dùng "
                "model theo kịch bản."
            )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover
            raise ModelNotConfigured("chưa cài langchain-openai") from exc

        self.name = f"openai:{model}"
        self._client = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            api_key=api_key,
        )

    def invoke(
        self,
        messages: Sequence[AnyMessage],
        *,
        tools: list[dict[str, Any]],
        config: dict[str, Any] | None = None,
    ) -> AIMessage:
        bound = self._client.bind_tools(tools) if tools else self._client
        # config được truyền tiếp để callback (ví dụ Langfuse) không bị cắt ở đây.
        response = bound.invoke(list(messages), config=config)
        if not isinstance(response, AIMessage):  # pragma: no cover - SDK đổi hành vi
            raise HarnessError(f"model trả về {type(response).__name__}, cần AIMessage")
        return response


# --------------------------------------------------------------------------- #
# Model theo kịch bản
# --------------------------------------------------------------------------- #
# Một bước trong kịch bản: hoặc là một AIMessage dựng sẵn, hoặc một hàm nhận
# (messages, tools) để quyết định theo ngữ cảnh - cần cho test luồng "model đọc lỗi
# rồi tự sửa".
ScriptStep = AIMessage | Callable[[list[AnyMessage], list[dict[str, Any]]], AIMessage]


@dataclass(slots=True)
class Invocation:
    """Ghi lại một lượt gọi, để test khẳng định harness đã đưa gì cho model."""

    message_count: int
    tool_names: list[str]
    last_message_type: str
    last_message_text: str


class ScriptedModelClient:
    """Model phát lại một kịch bản định trước. Hoàn toàn tất định."""

    def __init__(
        self,
        script: Sequence[ScriptStep],
        *,
        name: str = "scripted",
        on_exhausted: str = "error",
    ) -> None:
        if on_exhausted not in ("error", "finish"):
            raise ValueError("on_exhausted phải là 'error' hoặc 'finish'")
        self.name = name
        self._script = list(script)
        self._on_exhausted = on_exhausted
        self._cursor = 0
        self.invocations: list[Invocation] = []
        self._seen_message_ids: set[str] = set()
        self._seen_call_ids: set[str] = set()

    @property
    def calls_made(self) -> int:
        return self._cursor

    @property
    def exhausted(self) -> bool:
        return self._cursor >= len(self._script)

    def invoke(
        self,
        messages: Sequence[AnyMessage],
        *,
        tools: list[dict[str, Any]],
        config: dict[str, Any] | None = None,
    ) -> AIMessage:
        message_list = list(messages)
        last = message_list[-1] if message_list else None
        self.invocations.append(
            Invocation(
                message_count=len(message_list),
                tool_names=[t["function"]["name"] for t in tools],
                last_message_type=getattr(last, "type", "none"),
                last_message_text=str(getattr(last, "content", ""))[:400],
            )
        )

        if self.exhausted:
            if self._on_exhausted == "error":
                # Mặc định là lỗi: kịch bản hết nghĩa là test đã dự đoán sai số lượt,
                # và đó là thông tin cần thấy, không phải thứ cần che bằng một câu
                # trả lời mặc định.
                raise HarnessError(
                    f"kịch bản của ScriptedModelClient đã hết sau {self._cursor} lượt "
                    f"nhưng harness vẫn gọi tiếp (lượt {self._cursor + 1})"
                )
            return AIMessage(content="(kịch bản đã hết)")

        step = self._script[self._cursor]
        self._cursor += 1
        result = step(message_list, tools) if callable(step) else step
        if not isinstance(result, AIMessage):  # pragma: no cover - kịch bản viết sai
            raise HarnessError(f"bước kịch bản trả về {type(result).__name__}, cần AIMessage")
        return self._freshen(result)

    def _freshen(self, message: AIMessage) -> AIMessage:
        """Bảo đảm mỗi lượt trả về id mới, như một provider thật.

        Vì sao cần: một kịch bản mô phỏng "model lặp y nguyên một lời gọi" tự nhiên
        sẽ trả về **cùng một object** nhiều lần. Nhưng ``add_messages`` upsert theo
        ``id``, nên message thứ hai sẽ *ghi đè* message thứ nhất tại chỗ thay vì
        được nối thêm - lịch sử hội thoại không dài ra, và hai ToolMessage lại cùng
        trỏ về một ``tool_call_id``. Provider thật luôn sinh id mới cho mỗi response,
        nên test double phải làm đúng như vậy; nếu không, test sẽ kiểm một thế giới
        không tồn tại.

        Id do kịch bản đặt tường minh được giữ nguyên ở lần dùng đầu (test cần nó
        ổn định) và chỉ bị làm mới khi trùng với một id đã phát ra.
        """
        message_id = message.id
        if not message_id or message_id in self._seen_message_ids:
            message_id = f"ai_{self._cursor:02d}_{uuid.uuid4().hex[:8]}"
        self._seen_message_ids.add(message_id)

        tool_calls = []
        for index, raw in enumerate(message.tool_calls or []):
            call_id = raw.get("id")
            if not call_id or call_id in self._seen_call_ids:
                call_id = f"{call_id or 'call'}__t{self._cursor:02d}_{index}"
            self._seen_call_ids.add(call_id)
            tool_calls.append({"name": raw["name"], "args": dict(raw.get("args") or {}), "id": call_id})

        return AIMessage(
            content=message.content,
            tool_calls=tool_calls,
            id=message_id,
        )


# --------------------------------------------------------------------------- #
# Helper dựng kịch bản
# --------------------------------------------------------------------------- #
def say(text: str) -> AIMessage:
    """Lượt model trả lời bằng văn bản -> harness coi episode đã xong."""
    return AIMessage(content=text)


def call(name: str, call_id: str | None = None, **arguments: Any) -> dict[str, Any]:
    """Một lời gọi tool trong lượt assistant."""
    return {
        "name": name,
        "args": arguments,
        "id": call_id or f"call_{name}_{abs(hash(repr(sorted(arguments.items())))) % 10**6}",
    }


def calls(*tool_calls: dict[str, Any], text: str = "") -> AIMessage:
    """Lượt model xin một hoặc nhiều tool (nhiều tool = song song trong một lượt)."""
    return AIMessage(content=text, tool_calls=list(tool_calls))


def react_to_last_tool_result(
    predicate: Callable[[str], bool],
    if_true: ScriptStep,
    if_false: ScriptStep,
) -> ScriptStep:
    """Bước kịch bản rẽ nhánh theo nội dung ToolMessage gần nhất.

    Dùng để test luồng "model nhận lỗi rồi sửa": kịch bản phản ứng với thứ harness
    thật sự đã đưa cho model, chứ không giả định.
    """

    def step(messages: list[AnyMessage], tools: list[dict[str, Any]]) -> AIMessage:
        last_tool_text = ""
        for message in reversed(messages):
            if getattr(message, "type", "") == "tool":
                last_tool_text = str(message.content)
                break
        chosen = if_true if predicate(last_tool_text) else if_false
        return chosen(messages, tools) if callable(chosen) else chosen

    return step
