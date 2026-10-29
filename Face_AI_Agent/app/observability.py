"""Tracing qua Langfuse - tuỳ chọn, và tắt được hoàn toàn.

Khi bật, mỗi lần chạy graph là một trace và **mỗi node là một span mang đúng tên
đã đăng ký trong ``add_node``** (``validate_input``, ``detect_faces``, ...). Đó là lý
do tên node trong project này được đặt theo hành động và không đổi: đổi tên node là
đổi cấu trúc trace, làm vỡ mọi dashboard dựng trên tên span.

Cách gắn: handler đi qua ``config["callbacks"]``. LangGraph truyền ``config`` xuyên
qua mọi node và mọi Runnable lồng bên trong, nên chỉ cần gắn một lần ở ngoài. Hệ
quả cần nhớ: node nào tự tạo ``config`` mới thay vì nhận và truyền tiếp sẽ làm
**mất** phần trace bên dưới nó.

Langfuse không được cài hoặc thiếu key -> hàm này trả list rỗng và hệ thống chạy
bình thường. Tracing là thứ để quan sát, không phải thứ để phụ thuộc.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import Settings

logger = logging.getLogger(__name__)

_warned = False


def get_langfuse_callbacks(settings: Settings) -> list[Any]:
    """Trả về list callback handler (rỗng nếu tracing không khả dụng)."""
    global _warned

    if not settings.observability.langfuse_enabled:
        return []

    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        if not _warned:
            logger.warning(
                "langfuse_missing_keys",
                extra={"note": "langfuse_enabled=true nhưng thiếu LANGFUSE_PUBLIC_KEY/SECRET_KEY"},
            )
            _warned = True
        return []

    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        if not _warned:
            logger.warning(
                "langfuse_not_installed",
                extra={"note": "pip install langfuse để bật tracing, hoặc đặt langfuse_enabled=false"},
            )
            _warned = True
        return []

    try:
        return [CallbackHandler()]
    except Exception as exc:  # pragma: no cover - cấu hình langfuse sai
        if not _warned:
            logger.warning(
                "langfuse_init_failed",
                extra={"error_code": type(exc).__name__},
            )
            _warned = True
        return []


def build_run_config(
    settings: Settings,
    *,
    thread_id: str,
    request_id: str,
    user_id: str,
    extra_tags: list[str] | None = None,
) -> dict[str, Any]:
    """``RunnableConfig`` cho một lần chạy graph.

    ``thread_id`` là khoá checkpoint của LangGraph: cùng thread_id -> cùng một cuộc
    hội thoại, state được nạp lại từ checkpoint. ``request_id`` chỉ để trace, không
    ảnh hưởng tới persistence. Đừng lẫn hai cái.
    """
    return {
        "configurable": {"thread_id": thread_id},
        "callbacks": get_langfuse_callbacks(settings),
        "recursion_limit": settings.graph.recursion_limit,
        "run_name": "face_ai_agent",
        "tags": ["face-ai-agent", f"env:{settings.environment}"] + (extra_tags or []),
        "metadata": {
            "request_id": request_id,
            "thread_id": thread_id,
            # user_id đi vào metadata của trace để truy vết được theo người dùng.
            # KHÔNG đưa ảnh, embedding hay bất cứ dữ liệu sinh trắc học nào vào đây.
            "user_id": user_id,
            "langfuse_session_id": thread_id,
            "langfuse_user_id": user_id,
        },
    }
