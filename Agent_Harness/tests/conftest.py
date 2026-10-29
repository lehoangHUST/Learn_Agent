"""Fixture dùng chung.

Nguyên tắc của bộ test này: **model là thứ được mock, mọi thứ còn lại là thật.**
Harness, graph, checkpointer, registry, policy, hạn mức, guard SQL, database
SQLite, cổng phê duyệt - tất cả chạy thật. Nhờ vậy một test "model gọi tool không
tồn tại" thật sự kiểm chứng đường xử lý của harness, chứ không kiểm chứng một mock.

Database demo được tạo lại trong ``tmp_path`` cho từng test, bằng cùng hàm seed mà
người dùng chạy - nên test không bao giờ dùng một schema khác với thực tế.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.config import Settings
from app.wiring import build_runner
from harness.budget import BudgetPolicy
from harness.model import ScriptedModelClient, ScriptStep
from harness.registry import ToolRegistry
from harness.runner import HarnessRunner
from toolsets.sql.db import SqlLimits
from toolsets.sql.seed import build as build_demo_db
from toolsets.sql.tools import PERM_FINDINGS_WRITE, PERM_SQL_READ

USER = "test-user"


@pytest.fixture
def demo_db(tmp_path: Path) -> Path:
    path = tmp_path / "demo.db"
    build_demo_db(path, item_rows=400)
    return path


@pytest.fixture
def findings_path(tmp_path: Path) -> Path:
    return tmp_path / "findings.json"


@pytest.fixture
def settings(demo_db: Path, findings_path: Path) -> Settings:
    """Settings với hạn mức nhỏ, đặt tường minh.

    Hạn mức nhỏ là có chủ ý: test về chặn hạn mức không nên cần 24 lượt mới chạm
    ngưỡng. Và đặt tường minh ở đây nghĩa là test không phụ thuộc biến môi trường
    của máy đang chạy.
    """
    return Settings(
        db_path=demo_db,
        findings_path=findings_path,
        checkpointer="memory",
        model_provider="openai",
        openai_api_key=None,
        permissions=frozenset({PERM_SQL_READ, PERM_FINDINGS_WRITE}),
        budget=BudgetPolicy(
            max_model_calls=6,
            max_tool_calls=8,
            max_tool_calls_per_tool={"list_tables": 2},
            max_identical_tool_calls=2,
            max_active_seconds=30.0,
            max_tool_result_chars=1500,
            max_total_result_chars=8000,
            max_parallel_tool_calls=3,
        ),
        sql_limits=SqlLimits(max_rows=20, max_query_seconds=5.0, max_cell_chars=120),
        log_level="WARNING",
    )


@pytest.fixture
def make_runner(settings: Settings) -> Callable[..., HarnessRunner]:
    """Dựng runner với một kịch bản model cho trước.

    Mỗi runner có checkpointer riêng (``InMemorySaver``) nên các test không thấy
    thread của nhau.
    """

    def factory(
        script: Sequence[ScriptStep],
        *,
        registry: ToolRegistry | None = None,
        override: dict[str, Any] | None = None,
    ) -> HarnessRunner:
        local = settings
        if override:
            # Settings là dataclass slots -> tạo bản mới thay vì mutate, để test này
            # không ảnh hưởng test khác qua fixture dùng chung.
            import dataclasses

            local = dataclasses.replace(settings, **override)
        return build_runner(
            local,
            script=list(script),
            registry=registry,
            checkpointer=InMemorySaver(),
        )

    return factory


@pytest.fixture
def scripted_model() -> Callable[[Sequence[ScriptStep]], ScriptedModelClient]:
    def factory(script: Sequence[ScriptStep]) -> ScriptedModelClient:
        return ScriptedModelClient(list(script))

    return factory


# --------------------------------------------------------------------------- #
# Helper khẳng định
# --------------------------------------------------------------------------- #
def tool_steps(output: dict[str, Any], name: str | None = None) -> list[dict[str, Any]]:
    steps = [s for s in (output.get("steps") or []) if s.get("kind") == "tool"]
    return [s for s in steps if name is None or s.get("name") == name]


def tool_messages(runner: HarnessRunner, thread_id: str) -> list[Any]:
    """Các ToolMessage trong lịch sử hội thoại của một thread."""
    values = runner.get_state(thread_id).values or {}
    return [m for m in (values.get("messages") or []) if getattr(m, "type", "") == "tool"]


def assert_every_tool_call_answered(runner: HarnessRunner, thread_id: str) -> None:
    """Bất biến của giao thức tool-calling: mỗi tool_call_id có đúng một ToolMessage.

    Thiếu một cái thì lượt gọi model kế tiếp bị provider từ chối với một lỗi không
    nói gì về nguyên nhân thật - nên bất biến này được kiểm ở mọi test luồng, không
    chỉ ở một test riêng.
    """
    values = runner.get_state(thread_id).values or {}
    messages = values.get("messages") or []

    requested: list[str] = []
    for message in messages:
        if getattr(message, "type", "") == "ai":
            requested.extend(tc["id"] for tc in (message.tool_calls or []))
    answered = [m.tool_call_id for m in messages if getattr(message := m, "type", "") == "tool"]

    assert sorted(requested) == sorted(answered), (
        f"tool_call_id không khớp: xin {sorted(requested)}, trả {sorted(answered)}"
    )
