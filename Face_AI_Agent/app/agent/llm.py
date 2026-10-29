"""Bọc LLM: chỉ hai việc, và cả hai đều có đường chạy không cần LLM.

1. ``classify_intent`` - chỉ được gọi khi rule-based không chắc.
2. ``answer`` - diễn đạt lại ``grounding`` thành câu trả lời.

LLM là thành phần **tuỳ chọn**. Khi ``llm.enabled = false``, graph vẫn chạy đủ chức
năng: intent do rule quyết định, câu trả lời do template sinh ra từ cùng một
``grounding``. Con số trong hai chế độ là như nhau, vì cả hai đọc chung một nguồn.
Đây là lý do ``LLMError`` không bao giờ làm sập một request: node answer bắt lỗi và
rơi về template.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agent.prompts import (
    ANSWER_SYSTEM_PROMPT,
    INTENT_SYSTEM_PROMPT,
    build_answer_user_prompt,
    build_intent_user_prompt,
)
from app.config import Settings
from app.errors import LLMError, LLMNotConfigured

logger = logging.getLogger(__name__)


class IntentDecision(BaseModel):
    """Structured output của bước phân loại intent."""

    intent: Literal[
        "face_detection",
        "face_search",
        "face_detection_and_search",
        "result_question",
        "general_face_question",
        "unsupported_request",
    ] = Field(description="nhãn ý định")
    face_indices: list[int] | None = Field(
        default=None,
        description="chỉ số khuôn mặt (bắt đầu từ 0) nếu người dùng chỉ một mặt cụ thể",
    )
    reason: str = Field(default="", description="một câu giải thích vì sao chọn nhãn này")


class LLMClient:
    """Client mỏng quanh một chat model của LangChain."""

    def __init__(self, model: Any, settings: Settings) -> None:
        self._model = model
        self._settings = settings

    @classmethod
    def create(cls, settings: Settings) -> LLMClient:
        cfg = settings.llm
        if cfg.provider != "openai":
            raise LLMNotConfigured(
                f"provider {cfg.provider!r} chưa được hỗ trợ",
                details={"supported": ["openai"]},
            )
        if not settings.openai_api_key:
            raise LLMNotConfigured(
                "llm.enabled = true nhưng thiếu OPENAI_API_KEY. Đặt key, hoặc đặt "
                "llm.enabled = false để chạy ở chế độ xác định."
            )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - thiếu dependency
            raise LLMNotConfigured(
                "chưa cài langchain-openai. pip install langchain-openai, hoặc tắt LLM."
            ) from exc

        model = ChatOpenAI(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout_seconds,
            api_key=settings.openai_api_key,
        )
        return cls(model, settings)

    # ------------------------------------------------------------------ #
    def classify_intent(
        self,
        query: str,
        *,
        has_new_images: bool,
        prior_detection: bool,
        prior_search: bool,
        config: dict[str, Any] | None = None,
    ) -> IntentDecision:
        structured = self._model.with_structured_output(IntentDecision)
        messages = [
            ("system", INTENT_SYSTEM_PROMPT),
            (
                "human",
                build_intent_user_prompt(
                    query,
                    has_new_images=has_new_images,
                    prior_detection=prior_detection,
                    prior_search=prior_search,
                ),
            ),
        ]
        try:
            result = structured.invoke(messages, config=config)
        except Exception as exc:  # nhà cung cấp LLM có rất nhiều loại exception
            raise LLMError(f"classify_intent thất bại: {type(exc).__name__}: {exc}") from exc
        if not isinstance(result, IntentDecision):  # pragma: no cover - SDK đổi hành vi
            raise LLMError(f"structured output trả về {type(result).__name__}")
        return result

    def answer(
        self,
        query: str,
        grounding: dict[str, Any],
        *,
        config: dict[str, Any] | None = None,
    ) -> str:
        messages = [
            ("system", ANSWER_SYSTEM_PROMPT),
            ("human", build_answer_user_prompt(query, grounding)),
        ]
        try:
            response = self._model.invoke(messages, config=config)
        except Exception as exc:
            raise LLMError(f"answer thất bại: {type(exc).__name__}: {exc}") from exc

        content = getattr(response, "content", response)
        if isinstance(content, list):
            # Một số provider trả content dạng list các block.
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            content = "".join(parts) if parts else str(content)
        text = str(content).strip()
        if not text:
            raise LLMError("LLM trả về câu trả lời rỗng")
        return text

    def health(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "provider": self._settings.llm.provider,
            "model": self._settings.llm.model,
            "temperature": self._settings.llm.temperature,
            "use_for_intent": self._settings.llm.use_for_intent,
            "use_for_answer": self._settings.llm.use_for_answer,
        }
