"""Cấu hình - đọc từ biến môi trường, tường minh từng biến.

Không có file YAML ở project này: toàn bộ cấu hình là hạn mức và đường dẫn, số
lượng đủ nhỏ để một bảng env rõ ràng dễ đọc hơn một tầng merge.

Một ghi chú về giá trị mặc định của hạn mức: chúng được chọn **chặt**. Với harness,
mặc định rộng là một quyết định tốn tiền và khó phát hiện - episode vẫn "chạy được",
chỉ là mỗi câu hỏi tốn 30 lượt model. Mặc định chặt làm vấn đề lộ ra ngay dưới dạng
``stop_reason``, và nâng lên là một dòng config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from harness.budget import BudgetPolicy
from toolsets.sql.db import SqlLimits
from toolsets.sql.tools import PERM_FINDINGS_WRITE, PERM_SQL_READ

REPO_ROOT = Path(__file__).resolve().parent.parent


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def _path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw) if raw else default


@dataclass(slots=True)
class Settings:
    # --- dữ liệu ---
    db_path: Path = field(default_factory=lambda: _path("HARNESS_DB", REPO_ROOT / "data" / "demo.db"))
    findings_path: Path = field(
        default_factory=lambda: _path("HARNESS_FINDINGS", REPO_ROOT / "data" / "findings.json")
    )

    # --- checkpoint: bắt buộc có nếu muốn phê duyệt giữa episode ---
    checkpointer: str = field(default_factory=lambda: os.environ.get("HARNESS_CHECKPOINTER", "sqlite"))
    checkpoint_path: Path = field(
        default_factory=lambda: _path("HARNESS_CHECKPOINT_DB", REPO_ROOT / "data" / "checkpoints.db")
    )

    # --- model ---
    model_provider: str = field(default_factory=lambda: os.environ.get("HARNESS_MODEL_PROVIDER", "openai"))
    model_name: str = field(default_factory=lambda: os.environ.get("HARNESS_MODEL", "gpt-4o-mini"))
    model_temperature: float = field(default_factory=lambda: _float("HARNESS_TEMPERATURE", 0.0))
    openai_api_key: str | None = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY"))

    # --- quyền của principal ---
    # Mặc định cho bản demo: đọc được SQL và ghi được finding (ghi vẫn phải qua
    # phê duyệt). Bỏ findings:write ra là cách thử nhánh "model xin tool không có
    # quyền".
    permissions: frozenset[str] = field(
        default_factory=lambda: frozenset(
            (os.environ.get("HARNESS_PERMISSIONS") or f"{PERM_SQL_READ},{PERM_FINDINGS_WRITE}")
            .replace(" ", "")
            .split(",")
        )
        - {""}
    )

    # --- hạn mức ---
    budget: BudgetPolicy = field(
        default_factory=lambda: BudgetPolicy(
            max_model_calls=_int("HARNESS_MAX_MODEL_CALLS", 8),
            max_tool_calls=_int("HARNESS_MAX_TOOL_CALLS", 24),
            max_tool_calls_per_tool={
                "list_tables": _int("HARNESS_MAX_LIST_TABLES", 2),
                "describe_table": _int("HARNESS_MAX_DESCRIBE", 8),
                "run_query": _int("HARNESS_MAX_RUN_QUERY", 12),
                "profile_column": _int("HARNESS_MAX_PROFILE", 8),
                "save_finding": _int("HARNESS_MAX_SAVE_FINDING", 3),
            },
            max_identical_tool_calls=_int("HARNESS_MAX_IDENTICAL", 2),
            max_active_seconds=_float("HARNESS_MAX_ACTIVE_SECONDS", 120.0),
            max_tool_result_chars=_int("HARNESS_MAX_RESULT_CHARS", 4000),
            max_total_result_chars=_int("HARNESS_MAX_TOTAL_RESULT_CHARS", 40_000),
            max_parallel_tool_calls=_int("HARNESS_MAX_PARALLEL_TOOLS", 4),
        )
    )

    # --- giới hạn SQL ---
    sql_limits: SqlLimits = field(
        default_factory=lambda: SqlLimits(
            max_rows=_int("HARNESS_SQL_MAX_ROWS", 200),
            max_query_seconds=_float("HARNESS_SQL_TIMEOUT", 5.0),
            max_cell_chars=_int("HARNESS_SQL_MAX_CELL_CHARS", 200),
        )
    )

    log_level: str = field(default_factory=lambda: os.environ.get("HARNESS_LOG_LEVEL", "INFO"))

    def redacted(self) -> dict[str, object]:
        return {
            "db_path": str(self.db_path),
            "findings_path": str(self.findings_path),
            "checkpointer": self.checkpointer,
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "openai_api_key": "***set***" if self.openai_api_key else None,
            "permissions": sorted(self.permissions),
            "budget": {
                "max_model_calls": self.budget.max_model_calls,
                "max_tool_calls": self.budget.max_tool_calls,
                "max_identical_tool_calls": self.budget.max_identical_tool_calls,
                "max_active_seconds": self.budget.max_active_seconds,
                "max_parallel_tool_calls": self.budget.max_parallel_tool_calls,
            },
            "sql_limits": {
                "max_rows": self.sql_limits.max_rows,
                "max_query_seconds": self.sql_limits.max_query_seconds,
            },
        }


def load_settings() -> Settings:
    return Settings()
