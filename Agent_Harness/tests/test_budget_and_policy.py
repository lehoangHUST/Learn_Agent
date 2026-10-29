"""Test hạn mức, registry và policy ở mức đơn vị.

Những hàm này thuần (không I/O, không thời gian, không ngẫu nhiên) nên test được
trực tiếp bằng state dựng tay - nhanh và chỉ đúng nguyên nhân khi đỏ.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from harness.budget import (
    BudgetPolicy,
    count_identical_calls,
    tool_call_signature,
)
from harness.errors import ArgumentInvalid, DuplicateTool, ToolNotRegistered
from harness.policy import (
    DECISION_ALLOW,
    DECISION_DENY,
    DECISION_NEEDS_APPROVAL,
    cap_parallel_tool_calls,
    evaluate_tool_call,
)
from harness.registry import (
    SIDE_EFFECT_READ,
    SIDE_EFFECT_WRITE,
    ToolRegistry,
    ToolSpec,
)
from harness.state import (
    STOP_CONTEXT_BUDGET,
    STOP_DEADLINE,
    STOP_MAX_MODEL_CALLS,
    STOP_MAX_TOOL_CALLS,
)


class Args(BaseModel):
    model_config = {"extra": "forbid"}
    value: int = Field(ge=0)


def spec(**overrides: Any) -> ToolSpec:
    base = {
        "name": "demo",
        "description": "tool demo",
        "args_model": Args,
        "handler": lambda a, r: "ok",
    }
    return ToolSpec(**{**base, **overrides})


def state(**overrides: Any) -> dict[str, Any]:
    base = {
        "model_calls": 0,
        "tool_calls_total": 0,
        "tool_calls_by_name": {},
        "result_chars_total": 0,
        "elapsed_active_seconds": 0.0,
        "steps": [],
    }
    return {**base, **overrides}


# --------------------------------------------------------------------------- #
# Hạn mức
# --------------------------------------------------------------------------- #
def test_budget_checks_happen_before_spending() -> None:
    """Hạn mức kiểm TRƯỚC khi tiêu, nên lượt thứ N+1 không bao giờ được gọi."""
    budget = BudgetPolicy(max_model_calls=3)
    assert budget.check_before_model(state(model_calls=2)).allowed is True
    verdict = budget.check_before_model(state(model_calls=3))
    assert verdict.allowed is False
    assert verdict.stop_reason == STOP_MAX_MODEL_CALLS
    assert "3" in verdict.message


def test_tool_call_budget() -> None:
    budget = BudgetPolicy(max_tool_calls=2)
    assert budget.check_before_tool(state(tool_calls_total=1), "demo").allowed
    assert budget.check_before_tool(state(tool_calls_total=2), "demo").stop_reason == (
        STOP_MAX_TOOL_CALLS
    )


def test_deadline_counts_active_time_only() -> None:
    """Hạn mức thời gian dựa trên thời gian *làm việc*, không phải thời gian treo.

    Quan trọng với harness có phê duyệt: một lần chờ người duyệt 1 tiếng không được
    giết episode ngay khi nó được tiếp tục.
    """
    budget = BudgetPolicy(max_active_seconds=10.0)
    assert budget.check_before_model(state(elapsed_active_seconds=9.9)).allowed
    verdict = budget.check_before_model(state(elapsed_active_seconds=10.0))
    assert verdict.stop_reason == STOP_DEADLINE


def test_total_result_chars_budget() -> None:
    budget = BudgetPolicy(max_total_result_chars=1000)
    assert budget.check_before_tool(state(result_chars_total=999), "demo").allowed
    assert budget.check_before_tool(state(result_chars_total=1000), "demo").stop_reason == (
        STOP_CONTEXT_BUDGET
    )


def test_per_tool_quota_is_separate_from_episode_budget() -> None:
    """Hết quota một tool -> trả thông điệp; KHÔNG phải lý do dừng episode."""
    budget = BudgetPolicy(max_tool_calls=100, max_tool_calls_per_tool={"demo": 2})
    assert budget.tool_quota_exceeded(state(tool_calls_by_name={"demo": 1}), "demo") is None
    message = budget.tool_quota_exceeded(state(tool_calls_by_name={"demo": 2}), "demo")
    assert message is not None and "2 lần" in message
    # Tool khác không bị ảnh hưởng.
    assert budget.tool_quota_exceeded(state(tool_calls_by_name={"demo": 9}), "other") is None


def test_truncate_tells_the_model_how_much_was_dropped() -> None:
    budget = BudgetPolicy(max_tool_result_chars=50)
    text, dropped = budget.truncate("x" * 200)
    assert dropped == 150
    assert "150 ký tự nữa" in text
    # Phần giữ lại đúng bằng hạn mức; phần thông báo là thêm vào.
    assert text.startswith("x" * 50)


def test_truncate_leaves_small_results_alone() -> None:
    budget = BudgetPolicy(max_tool_result_chars=50)
    text, dropped = budget.truncate("ngắn")
    assert (text, dropped) == ("ngắn", 0)


def test_signature_ignores_key_order() -> None:
    """Chữ ký phải ổn định: thứ tự key do model sinh không làm hai lời gọi trông khác."""
    assert tool_call_signature("t", {"a": 1, "b": 2}) == tool_call_signature("t", {"b": 2, "a": 1})
    assert tool_call_signature("t", {"a": 1}) != tool_call_signature("t", {"a": 2})
    assert tool_call_signature("t1", {"a": 1}) != tool_call_signature("t2", {"a": 1})


def test_count_identical_calls_reads_the_transcript() -> None:
    signature = tool_call_signature("demo", {"value": 1})
    steps = [
        {"kind": "tool", "detail": {"signature": signature}},
        {"kind": "model", "detail": {}},
        {"kind": "tool", "detail": {"signature": signature}},
        {"kind": "tool", "detail": {"signature": "khác"}},
    ]
    assert count_identical_calls(state(steps=steps), signature) == 2


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_registry_rejects_duplicates() -> None:
    registry = ToolRegistry([spec()])
    with pytest.raises(DuplicateTool):
        registry.register(spec())


def test_registry_get_unknown_is_a_harness_error() -> None:
    """Code hỏi sai tool là lỗi harness; *model* hỏi sai tool thì không (xem loop)."""
    with pytest.raises(ToolNotRegistered):
        ToolRegistry().get("khong-co")


def test_write_tool_must_require_approval() -> None:
    with pytest.raises(ValueError, match="requires_approval"):
        spec(side_effect=SIDE_EFFECT_WRITE, requires_approval=False)


def test_invalid_side_effect_rejected() -> None:
    with pytest.raises(ValueError, match="side_effect"):
        spec(side_effect="maybe")


def test_argument_validation_error_includes_schema_for_the_model() -> None:
    """Lỗi tham số phải kèm schema: model cần đủ thông tin để sửa ở lượt sau."""
    with pytest.raises(ArgumentInvalid) as exc:
        spec().validate_arguments({"value": -1})
    assert exc.value.recoverable is True
    assert "properties" in (exc.value.hint or "")


def test_extra_arguments_are_rejected() -> None:
    """``extra='forbid'``: model bịa thêm tham số thì phải biết, không im lặng bỏ qua."""
    with pytest.raises(ArgumentInvalid):
        spec().validate_arguments({"value": 1, "bonus": True})


def test_non_dict_arguments_rejected() -> None:
    with pytest.raises(ArgumentInvalid):
        spec().validate_arguments("không phải dict")  # type: ignore[arg-type]


def test_openai_tool_shape() -> None:
    tool = spec().to_openai_tool()
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "demo"
    assert "properties" in tool["function"]["parameters"]


def test_catalog_marks_risky_tools() -> None:
    """System prompt phải nói rõ tool nào cần phê duyệt."""
    registry = ToolRegistry(
        [
            spec(name="doc", side_effect=SIDE_EFFECT_READ),
            spec(name="ghi", side_effect=SIDE_EFFECT_WRITE, requires_approval=True),
        ]
    )
    catalog = registry.catalog_text()
    assert "- doc:" in catalog
    assert "GHI, CẦN PHÊ DUYỆT" in catalog


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #
def test_policy_allows_plain_read_tool() -> None:
    decision = evaluate_tool_call(
        spec(),
        {"value": 1},
        state=state(),
        permissions=frozenset(),
        budget=BudgetPolicy(),
    )
    assert decision.decision == DECISION_ALLOW


def test_policy_requires_permission() -> None:
    decision = evaluate_tool_call(
        spec(required_permission="sql:read"),
        {"value": 1},
        state=state(),
        permissions=frozenset({"other"}),
        budget=BudgetPolicy(),
    )
    assert decision.decision == DECISION_DENY
    assert "sql:read" in decision.model_message


def test_policy_checks_permission_before_approval() -> None:
    """Thiếu quyền thì không ai bị hỏi phê duyệt - chặn ở bước rẻ và dứt khoát trước."""
    decision = evaluate_tool_call(
        spec(
            side_effect=SIDE_EFFECT_WRITE,
            requires_approval=True,
            required_permission="w",
        ),
        {"value": 1},
        state=state(),
        permissions=frozenset(),
        budget=BudgetPolicy(),
    )
    assert decision.decision == DECISION_DENY  # không phải NEEDS_APPROVAL


def test_policy_denies_identical_repeat() -> None:
    signature = tool_call_signature("demo", {"value": 1})
    steps = [{"kind": "tool", "detail": {"signature": signature}}] * 2
    decision = evaluate_tool_call(
        spec(),
        {"value": 1},
        state=state(steps=steps),
        permissions=frozenset(),
        budget=BudgetPolicy(max_identical_tool_calls=2),
    )
    assert decision.decision == DECISION_DENY
    assert "đã chạy 2 lần" in decision.model_message
    # Thông điệp phải nói model NÊN LÀM GÌ, không chỉ nói nó sai.
    assert "đổi cách tiếp cận" in (decision.hint or "")


def test_policy_needs_approval_for_write() -> None:
    decision = evaluate_tool_call(
        spec(side_effect=SIDE_EFFECT_WRITE, requires_approval=True, risk="ghi dữ liệu"),
        {"value": 1},
        state=state(),
        permissions=frozenset(),
        budget=BudgetPolicy(),
    )
    assert decision.decision == DECISION_NEEDS_APPROVAL
    assert decision.risk == "ghi dữ liệu"


def test_policy_does_not_look_at_tool_output() -> None:
    """Bất biến chống prompt injection, phát biểu ở mức chữ ký hàm.

    ``evaluate_tool_call`` chỉ nhận spec, tham số, state, quyền và hạn mức. Nó
    **không có tham số nào** mang nội dung kết quả tool. Test này khoá lại điều đó:
    ai thêm một tham số như vậy vào sẽ làm test đỏ, và sẽ phải giải thích vì sao
    output không đáng tin lại được tham gia quyết định.
    """
    import inspect

    parameters = set(inspect.signature(evaluate_tool_call).parameters)
    assert parameters == {"spec", "arguments", "state", "permissions", "budget"}


# --------------------------------------------------------------------------- #
# Giới hạn song song
# --------------------------------------------------------------------------- #
def test_cap_parallel_splits_calls() -> None:
    calls_list = [{"id": str(i)} for i in range(6)]
    run, deferred = cap_parallel_tool_calls(calls_list, BudgetPolicy(max_parallel_tool_calls=2))
    assert len(run) == 2
    assert len(deferred) == 4
    # Không mất lời gọi nào: mỗi tool_call_id vẫn phải nhận một ToolMessage.
    assert len(run) + len(deferred) == len(calls_list)


def test_cap_parallel_always_runs_at_least_one() -> None:
    """Cấu hình 0 không được làm episode bế tắc hoàn toàn."""
    run, deferred = cap_parallel_tool_calls(
        [{"id": "a"}, {"id": "b"}], BudgetPolicy(max_parallel_tool_calls=0)
    )
    assert len(run) == 1
    assert len(deferred) == 1
