"""Schema cho ``/agent/chat`` và ``/tasks/{request_id}``."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=4000,
        description="câu hỏi/yêu cầu bằng ngôn ngữ tự nhiên",
    )
    images_base64: list[str] | None = Field(
        default=None,
        max_length=16,
        description="ảnh kèm theo (tuỳ chọn). Không có ảnh thì agent dùng kết quả đã lưu của thread.",
    )
    thread_id: str | None = Field(
        default=None,
        description=(
            "định danh cuộc hội thoại. Truyền lại cùng thread_id để hỏi tiếp về kết quả "
            "trước đó. Bỏ trống -> hệ thống sinh mới và trả về trong response."
        ),
    )
    face_indices: list[int] | None = Field(
        default=None,
        description=(
            "chỉ định khuôn mặt cụ thể (0-based). Nếu bỏ trống, agent tự suy từ câu "
            'nói ("khuôn mặt thứ hai" -> [1]).'
        ),
    )
    target_result_id: str | None = Field(
        default=None,
        description="hỏi về một kết quả cụ thể thay vì kết quả mới nhất của thread",
    )
    collection: str | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"message": "Có bao nhiêu khuôn mặt trong ảnh?", "images_base64": ["<base64>"]},
                {"message": "Tìm kiếm khuôn mặt thứ hai", "thread_id": "thr_1a2b3c"},
                {"message": "Kết quả thứ hai có similarity bao nhiêu?", "thread_id": "thr_1a2b3c"},
            ]
        }
    }


class ErrorEntryOut(BaseModel):
    code: str
    message: str
    retryable: bool
    node: str
    details: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    request_id: str
    thread_id: str = Field(description="truyền lại ở lượt sau để hỏi tiếp")
    status: Literal["succeeded", "partial", "failed"]
    intent: str
    intent_source: Literal["rule", "llm", "fallback"] = Field(
        description="rule = phân loại tất định; llm = do model quyết định"
    )
    answer: str = Field(description="câu trả lời. Mọi số trong đây đều truy ngược được về dữ liệu.")
    answer_source: Literal["template", "llm"] = Field(
        description="template = sinh tất định từ dữ liệu; llm = do model diễn đạt lại"
    )
    detection_result_id: str | None = None
    search_result_id: str | None = None
    detection_summary: dict[str, Any] | None = None
    search_summary: dict[str, Any] | None = None
    analysis: dict[str, Any] | None = None
    executed_steps: list[str] = Field(
        default_factory=list, description="các node đã chạy, theo thứ tự"
    )
    errors: list[ErrorEntryOut] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TaskStatusResponse(BaseModel):
    """Trạng thái của một lượt đã chạy.

    Hệ thống này xử lý **đồng bộ**: ``/agent/chat`` trả kết quả ngay. Endpoint này
    phục vụ việc tra lại kết quả theo ``request_id`` - hữu ích khi client mất response,
    và là chỗ sẵn sàng cho chế độ bất đồng bộ về sau (khi đó ``status`` sẽ có thêm
    ``queued``/``running``).
    """

    request_id: str
    status: Literal["succeeded", "partial", "failed", "not_found"]
    thread_id: str | None = None
    results: list[dict[str, Any]] = Field(
        default_factory=list, description="các kết quả gắn với request_id này"
    )
    note: str | None = None


class ThreadResultsResponse(BaseModel):
    thread_id: str
    results: list[dict[str, Any]]


class ExplainResponse(BaseModel):
    result_id: str
    threshold: dict[str, Any]
    threshold_calibrated: bool
    index: dict[str, Any] | None = None
    explanations: list[dict[str, Any]]
    score_semantics: str
    caveat: str
