"""Chạy graph: điểm vào duy nhất cho cả API và test.

Việc của runner:

1. Validate ảnh và nạp chúng vào ``ImageBuffer`` (trong RAM, chỉ sống trong lần chạy).
2. Dựng ``FaceAgentContext`` - principal, collection, dependency, buffer ảnh.
3. Dựng ``RunnableConfig`` - ``thread_id`` cho checkpoint, callbacks cho Langfuse.
4. Gọi ``graph.invoke`` và trả về output đã theo ``FaceAgentOutput``.

Vì sao tách khỏi tầng API: cùng một đường chạy được dùng bởi FastAPI, bởi test, và
bởi script CLI. Nếu API tự lắp context thì test sẽ lắp theo cách hơi khác, và đến
lúc nào đó sự khác biệt đó chính là chỗ bug ẩn.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage

from app.agent.state import (
    FaceAgentContext,
    ImageBuffer,
    ImageRef,
    initial_state,
)
from app.dependencies import AgentDeps
from app.logging_utils import new_request_id, new_thread_id
from app.observability import build_run_config
from app.security import Principal
from app.vision.preprocessing import ValidatedImage

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AgentRunResult:
    output: dict[str, Any]
    request_id: str
    thread_id: str

    @property
    def status(self) -> str:
        return str(self.output.get("status", "unknown"))

    @property
    def answer(self) -> str:
        return str(self.output.get("answer", ""))


class FaceAgentRunner:
    def __init__(self, graph: Any, deps: AgentDeps) -> None:
        self._graph = graph
        self._deps = deps

    @property
    def deps(self) -> AgentDeps:
        return self._deps

    def run(
        self,
        *,
        principal: Principal,
        query: str,
        images_base64: list[str] | None = None,
        images: list[ValidatedImage] | None = None,
        collection: str | None = None,
        thread_id: str | None = None,
        request_id: str | None = None,
        face_indices: list[int] | None = None,
        target_result_id: str | None = None,
    ) -> AgentRunResult:
        """Chạy một lượt.

        Truyền ``images_base64`` (API) hoặc ``images`` đã validate sẵn (test/CLI).
        """
        deps = self._deps
        collection = collection or deps.settings.vector_store.collection
        request_id = request_id or new_request_id()
        thread_id = thread_id or new_thread_id()

        validated: list[ValidatedImage] = list(images or [])
        if images_base64:
            # Validate TRƯỚC khi vào graph: ảnh sai thì phải lỗi 4xx ngay, không phải
            # đi hết graph rồi mới báo.
            validated.extend(deps.detection.validate_many(images_base64))

        buffer = ImageBuffer()
        image_refs: list[ImageRef] = [
            buffer.add("imgref_" + uuid.uuid4().hex[:16], image) for image in validated
        ]

        context = FaceAgentContext(
            principal=principal,
            collection=collection,
            deps=deps,
            images=buffer,
        )

        payload = dict(
            initial_state(
                query=query,
                request_id=request_id,
                thread_id=thread_id,
                user_id=principal.user_id,
                image_refs=image_refs,
                requested_face_indices=face_indices,
                target_result_id=target_result_id,
            )
        )
        # Lưu câu của người dùng vào lịch sử hội thoại. Lịch sử này dùng cho ngữ cảnh
        # đối thoại; dữ liệu kết quả thì KHÔNG lấy từ đây mà từ result store.
        payload["messages"] = [HumanMessage(content=query)] if query else []

        config = build_run_config(
            deps.settings,
            thread_id=thread_id,
            request_id=request_id,
            user_id=principal.user_id,
        )

        logger.info(
            "agent_run_start",
            extra={
                "request_id": request_id,
                "thread_id": thread_id,
                "image_count": len(image_refs),
                "collection": collection,
                **principal.log_fields(),
            },
        )

        output = self._graph.invoke(payload, config, context=context)

        logger.info(
            "agent_run_end",
            extra={
                "request_id": request_id,
                "thread_id": thread_id,
                "status": output.get("status"),
                "intent": output.get("intent"),
                "error_count": len(output.get("errors") or []),
                "executed_steps": output.get("executed_steps"),
            },
        )
        return AgentRunResult(output=output, request_id=request_id, thread_id=thread_id)

    def get_state(self, thread_id: str) -> Any:
        """State đã checkpoint của một thread. Dùng để debug và time-travel."""
        return self._graph.get_state({"configurable": {"thread_id": thread_id}})
