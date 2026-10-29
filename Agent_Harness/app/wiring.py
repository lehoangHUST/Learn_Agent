"""Lắp các thành phần lại: model + registry + budget + database -> runner.

Đây cũng là chỗ thể hiện ranh giới của harness: ``harness/`` không import gì từ
``toolsets/``, và ``toolsets/`` không import gì từ ``harness/loop.py``. Chúng chỉ
gặp nhau ở file này. Hệ quả thực tế: đổi toolset (từ SQL sang file system, sang API
nội bộ) không phải sửa một dòng nào trong harness.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable, Sequence
from typing import Any

from app.config import Settings
from harness.errors import ModelNotConfigured
from harness.loop import build_harness
from harness.model import ModelClient, OpenAIModelClient, ScriptedModelClient, ScriptStep
from harness.registry import ToolRegistry
from harness.runner import HarnessRunner
from harness.state import HarnessContext
from toolsets.sql.db import ReadOnlyDatabase
from toolsets.sql.tools import build_sql_registry

logger = logging.getLogger(__name__)


def build_model(settings: Settings, *, script: Sequence[ScriptStep] | None = None) -> ModelClient:
    """Chọn model.

    ``script`` khác None -> dùng model theo kịch bản, bất kể cấu hình. Dùng cho test
    và cho demo offline. Không có đường nào để một kịch bản lọt vào production mà
    không ai gọi tường minh.
    """
    if script is not None:
        return ScriptedModelClient(script)
    if settings.model_provider == "scripted":
        raise ModelNotConfigured(
            "model_provider='scripted' nhưng không có kịch bản nào được truyền vào"
        )
    if settings.model_provider != "openai":
        raise ModelNotConfigured(
            f"provider {settings.model_provider!r} chưa hỗ trợ (chỉ có 'openai')"
        )
    return OpenAIModelClient(
        settings.model_name,
        api_key=settings.openai_api_key,
        temperature=settings.model_temperature,
    )


def build_checkpointer(settings: Settings) -> Any | None:
    """Checkpointer.

    **Bắt buộc phải có** nếu muốn dùng phê duyệt: không có checkpoint thì không có
    chỗ lưu trạng thái lúc ``interrupt()``, và episode không thể tiếp tục. Chế độ
    ``none`` chỉ dùng được cho episode toàn tool đọc.
    """
    if settings.checkpointer == "none":
        logger.warning(
            "checkpointer_disabled",
            extra={"note": "không có checkpoint -> không dùng được phê duyệt giữa episode"},
        )
        return None
    if settings.checkpointer == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()

    from langgraph.checkpoint.sqlite import SqliteSaver

    settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.checkpoint_path), check_same_thread=False)
    return SqliteSaver(conn)


def build_context_factory(
    settings: Settings,
    model: ModelClient,
    registry: ToolRegistry | None = None,
) -> Callable[[str], HarnessContext]:
    """Trả về hàm dựng context cho mỗi lần invoke.

    Là *factory* chứ không phải một context dựng sẵn vì context mang ``user_id``, và
    vì nó phải được dựng lại ở mỗi lần invoke (kể cả lần resume sau phê duyệt) -
    không thứ gì trong nó đi qua checkpoint.
    """
    registry = registry or build_sql_registry()
    database = ReadOnlyDatabase(settings.db_path, settings.sql_limits)

    def factory(user_id: str) -> HarnessContext:
        return HarnessContext(
            model=model,
            registry=registry,
            budget=settings.budget,
            user_id=user_id,
            permissions=settings.permissions,
            tool_context={
                "database": database,
                "findings_path": settings.findings_path,
            },
        )

    return factory


def build_runner(
    settings: Settings,
    *,
    script: Sequence[ScriptStep] | None = None,
    model: ModelClient | None = None,
    registry: ToolRegistry | None = None,
    checkpointer: Any | None = None,
) -> HarnessRunner:
    model = model or build_model(settings, script=script)
    graph = build_harness(
        checkpointer=checkpointer if checkpointer is not None else build_checkpointer(settings)
    )
    logger.info(
        "harness_ready",
        extra={"model": model.name, "permissions": sorted(settings.permissions)},
    )
    return HarnessRunner(graph, build_context_factory(settings, model, registry))
