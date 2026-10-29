"""Vòng lặp agent - cấu trúc định nghĩa nên một harness.

::

    START -> prepare -> call_model -+--(model xin tool)--> execute_tools --+
                            ^       |                                     |
                            |       +--(model trả lời)----> finish -> END  |
                            +-------------------------------------------- +

Khác biệt cốt lõi so với một workflow: **không có kế hoạch nào được tính trước.**
Cạnh ``call_model -> execute_tools`` được chọn hay không phụ thuộc vào việc model
có đặt ``tool_calls`` trong lượt vừa rồi - một dữ kiện chỉ tồn tại lúc chạy. Số
vòng lặp cũng vậy. Harness không biết episode sẽ đi qua bao nhiêu super-step; nó
chỉ biết **giới hạn** của số đó.

Ba chi tiết mà một harness làm sai thì sẽ sai âm thầm
----------------------------------------------------
**1. Mỗi ``tool_call_id`` phải có đúng một ``ToolMessage``.** Giao thức tool-calling
yêu cầu vậy. Thiếu một cái thì lượt gọi model tiếp theo bị provider từ chối với một
lỗi không nói gì về nguyên nhân thật. Nên mọi nhánh - tool không tồn tại, tham số
sai, bị policy chặn, bị người từ chối, bị hoãn vì quá nhiều tool song song - đều
*vẫn* sinh một ToolMessage.

**2. Phê duyệt phải xảy ra TRƯỚC khi bất kỳ tool nào trong lượt đó chạy.**
``interrupt()`` làm node dừng và **chạy lại từ đầu** khi được resume. Nếu node thực
thi tuần tự "tool 1 chạy, tool 2 cần phê duyệt -> interrupt", thì khi resume tool 1
sẽ chạy **lần thứ hai**. Với một tool đọc thì đó là lãng phí; với một tool ghi thì
đó là ghi hai lần. Vì vậy node chia hai pha rạch ròi: pha *lập kế hoạch* (thuần -
validate tham số, hỏi policy, không chạy gì), rồi một lần ``interrupt()`` duy nhất
cho mọi lời gọi cần phê duyệt, rồi pha *thực thi*.

**3. ``GraphInterrupt`` không phải lỗi.** Nó là cơ chế tạm dừng của LangGraph. Một
``except Exception`` đặt quanh thân node sẽ nuốt nó và biến một lần chờ phê duyệt
thành một lỗi tool. Code dưới đây bắt nó lại và ``raise`` tiếp, có chú thích, vì
đây là loại bug rất khó thấy khi đọc.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from harness.budget import tool_call_signature
from harness.errors import ApprovalDenied, HarnessError, PolicyDenied, ToolFailure
from harness.policy import cap_parallel_tool_calls, evaluate_tool_call
from harness.prompts import STOPPED_ANSWER_TEMPLATE, build_system_prompt
from harness.registry import ToolRuntime, ToolSpec
from harness.state import (
    STATUS_ANSWERED,
    STATUS_RUNNING,
    STATUS_STOPPED,
    STOP_ANSWERED,
    HarnessContext,
    HarnessInput,
    HarnessOutput,
    HarnessState,
    PendingApproval,
    StepRecord,
    fresh_counters,
)

logger = logging.getLogger(__name__)

NODE_PREPARE = "prepare"
NODE_CALL_MODEL = "call_model"
NODE_EXECUTE_TOOLS = "execute_tools"
NODE_FINISH = "finish"


# --------------------------------------------------------------------------- #
# prepare
# --------------------------------------------------------------------------- #
def prepare(state: HarnessState, runtime: Runtime[HarnessContext]) -> dict[str, Any]:
    """Dựng tin nhắn mở đầu và đặt lại bộ đếm.

    Reset ở đây chứ không ở input: ``input_schema`` hẹp nên LangGraph **lọc bỏ im
    lặng** mọi key ngoài schema đó - một reset gửi từ ngoài sẽ không có tác dụng và
    cũng không báo lỗi. Node này cũng không thể lỗi, nên không có nhánh nào bỏ sót
    việc reset.
    """
    ctx = runtime.context
    goal = (state.get("goal") or "").strip()
    if not goal:
        raise HarnessError("episode cần một goal không rỗng")

    system = build_system_prompt(ctx.registry, ctx.budget)
    return {
        **fresh_counters(),
        "messages": [SystemMessage(content=system), HumanMessage(content=goal)],
        "steps": [
            StepRecord(
                index=0,
                kind="model",
                name="prepare",
                status="ok",
                duration_ms=0.0,
                detail={
                    "tools_offered": ctx.registry.names(),
                    "budget": {
                        "max_model_calls": ctx.budget.max_model_calls,
                        "max_tool_calls": ctx.budget.max_tool_calls,
                        "max_active_seconds": ctx.budget.max_active_seconds,
                    },
                },
            )
        ],
    }


# --------------------------------------------------------------------------- #
# call_model
# --------------------------------------------------------------------------- #
def call_model(state: HarnessState, runtime: Runtime[HarnessContext]) -> dict[str, Any]:
    """Gọi model. Kiểm hạn mức **trước** khi tiêu một lượt gọi."""
    ctx = runtime.context
    verdict = ctx.budget.check_before_model(state)
    if not verdict.allowed:
        logger.info(
            "episode_stopped",
            extra={
                "episode_id": state.get("episode_id"),
                "stop_reason": verdict.stop_reason,
                "model_calls": state.get("model_calls"),
            },
        )
        return {
            "status": STATUS_STOPPED,
            "stop_reason": verdict.stop_reason,
            "pending_tool_calls": [],
            "steps": [
                _step(state, "stop", verdict.stop_reason or "stopped", "error", 0.0,
                      {"message": verdict.message})
            ],
        }

    started = time.perf_counter()
    tools = ctx.registry.openai_tools()
    response = ctx.model.invoke(state["messages"], tools=tools, config=None)
    elapsed = (time.perf_counter() - started) * 1000

    requested = [tc.get("name") for tc in (response.tool_calls or [])]
    logger.info(
        "model_turn",
        extra={
            "episode_id": state.get("episode_id"),
            "model": ctx.model.name,
            "turn": state.get("model_calls", 0) + 1,
            "tool_calls_requested": requested,
            "latency_ms": round(elapsed, 2),
        },
    )

    return {
        "messages": [response],
        # Tín hiệu điều khiển tường minh - xem ghi chú ở `pending_tool_calls`
        # trong state.py về việc vì sao không suy từ message cuối cùng.
        "pending_tool_calls": [dict(tc) for tc in (response.tool_calls or [])],
        "model_calls": state.get("model_calls", 0) + 1,
        "elapsed_active_seconds": state.get("elapsed_active_seconds", 0.0) + (elapsed / 1000),
        "steps": [
            _step(
                state,
                "model",
                ctx.model.name,
                "ok",
                elapsed,
                {
                    "tool_calls_requested": requested,
                    "content_chars": len(str(response.content or "")),
                },
            )
        ],
    }


# --------------------------------------------------------------------------- #
# execute_tools
# --------------------------------------------------------------------------- #
def execute_tools(state: HarnessState, runtime: Runtime[HarnessContext]) -> dict[str, Any]:
    """Chạy các tool model vừa xin, theo hai pha: lập kế hoạch rồi thực thi.

    Xem docstring module về việc vì sao hai pha là bắt buộc chứ không phải cho gọn.
    """
    ctx = runtime.context
    requested = list(state.get("pending_tool_calls") or [])
    if not requested:  # pragma: no cover - router đã kiểm
        raise HarnessError("execute_tools được gọi nhưng không có lời gọi tool nào")

    to_run, deferred = cap_parallel_tool_calls(requested, ctx.budget)

    # ---- PHA 1: lập kế hoạch. Thuần, không side effect, chạy lại được an toàn. ----
    plans: list[dict[str, Any]] = []
    needs_approval: list[PendingApproval] = []

    for tool_call in to_run:
        plan = _plan_one(tool_call, state=state, ctx=ctx)
        plans.append(plan)
        if plan["outcome"] == "needs_approval":
            needs_approval.append(plan["approval"])

    # ---- Một lần interrupt duy nhất cho cả lượt ----
    decisions: dict[str, dict[str, Any]] = {}
    if needs_approval:
        # interrupt() dừng graph ở đây. Khi resume, TOÀN BỘ node chạy lại từ đầu,
        # nên pha 1 phải không có side effect - và nó không có.
        raw = interrupt(
            {
                "type": "tool_approval",
                "episode_id": state.get("episode_id"),
                "goal": state.get("goal"),
                "requests": list(needs_approval),
            }
        )
        decisions = _normalize_decisions(raw, needs_approval)

    # ---- PHA 2: thực thi ----
    messages: list[AnyMessage] = []
    steps: list[StepRecord] = []
    counts = dict(state.get("tool_calls_by_name") or {})
    total = state.get("tool_calls_total", 0)
    chars = state.get("result_chars_total", 0)
    elapsed_total = state.get("elapsed_active_seconds", 0.0)
    stop_reason: str | None = None

    for plan in plans:
        tool_call_id = plan["tool_call_id"]
        name = plan["name"]

        # Hạn mức tổng được kiểm lại ở đây: lượt assistant này có thể xin 4 tool và
        # tool thứ 3 mới là cái vượt hạn mức.
        verdict = ctx.budget.check_before_tool({**state, "tool_calls_total": total}, name)
        if not verdict.allowed:
            stop_reason = verdict.stop_reason
            messages.append(
                _tool_error_message(
                    tool_call_id,
                    name,
                    PolicyDenied(verdict.message, code=verdict.stop_reason or "budget"),
                )
            )
            steps.append(_step(state, "tool", name, "denied", 0.0, {"reason": verdict.stop_reason}))
            continue

        if plan["outcome"] == "failed":
            failure: ToolFailure = plan["failure"]
            messages.append(_tool_error_message(tool_call_id, name, failure))
            steps.append(
                _step(state, "tool", name, "error", 0.0, {"code": failure.code})
            )
            # Lời gọi không chạy thì không tính vào hạn mức tool: hạn mức đó bảo vệ
            # tài nguyên, còn đây là một lỗi mà model cần được phép sửa.
            continue

        if plan["outcome"] == "needs_approval":
            decision = decisions.get(tool_call_id, {"approved": False, "reason": "không có quyết định"})
            if not decision.get("approved"):
                reason = str(decision.get("reason") or "người dùng từ chối")
                messages.append(
                    _tool_error_message(
                        tool_call_id,
                        name,
                        ApprovalDenied(f"người dùng không phê duyệt: {reason}"),
                    )
                )
                steps.append(_step(state, "tool", name, "denied", 0.0, {"reason": reason}))
                continue
            steps.append(
                _step(state, "approval", name, "ok", 0.0, {"approved_by": state.get("user_id")})
            )

        # --- chạy thật ---
        started = time.perf_counter()
        try:
            raw_result = plan["spec"].handler(
                plan["arguments"],
                ToolRuntime(user_id=state.get("user_id", ""), context=ctx.tool_context),
            )
            text = raw_result if isinstance(raw_result, str) else str(raw_result)
            status = "ok"
            failure = None
        except GraphInterrupt:
            # KHÔNG nuốt: đây là cơ chế tạm dừng của LangGraph, không phải lỗi tool.
            raise
        except ToolFailure as exc:
            text, status, failure = exc.to_tool_content(), "error", exc
        except Exception as exc:
            logger.exception(
                "tool_crashed",
                extra={"episode_id": state.get("episode_id"), "tool": name},
            )
            failure = ToolFailure(
                f"tool {name} lỗi không lường trước: {type(exc).__name__}",
                code="tool_crashed",
                recoverable=True,
                hint="Thử lại với tham số đơn giản hơn, hoặc dùng tool khác.",
            )
            text, status = failure.to_tool_content(), "error"

        duration = (time.perf_counter() - started) * 1000
        elapsed_total += duration / 1000
        truncated_text, dropped = ctx.budget.truncate(text)

        messages.append(
            ToolMessage(
                content=truncated_text,
                tool_call_id=tool_call_id,
                name=name,
                status="error" if status == "error" else "success",
            )
        )
        total += 1
        counts[name] = counts.get(name, 0) + 1
        chars += len(truncated_text)
        steps.append(
            _step(
                state,
                "tool",
                name,
                status,
                duration,
                {
                    "signature": plan["signature"],
                    "result_chars": len(truncated_text),
                    "truncated_chars": dropped,
                    "code": getattr(failure, "code", None),
                },
            )
        )

    # Lời gọi bị hoãn vì vượt giới hạn song song: vẫn phải có ToolMessage cho mỗi id.
    for tool_call in deferred:
        messages.append(
            _tool_error_message(
                tool_call["id"],
                tool_call.get("name", "?"),
                PolicyDenied(
                    f"lượt này đã xin {len(requested)} tool, vượt giới hạn "
                    f"{ctx.budget.max_parallel_tool_calls} tool/lượt",
                    hint="Hãy xin ít tool hơn mỗi lượt; lời gọi này chưa chạy.",
                ),
            )
        )
        steps.append(
            _step(state, "tool", tool_call.get("name", "?"), "skipped", 0.0,
                  {"reason": "parallel_limit"})
        )

    update: dict[str, Any] = {
        "messages": messages,
        "tool_calls_total": total,
        "tool_calls_by_name": counts,
        "result_chars_total": chars,
        "elapsed_active_seconds": elapsed_total,
        "steps": steps,
        "pending_approval": None,
        # Lượt này đã xử lý xong; xoá tín hiệu để lượt model kế tiếp tự đặt lại.
        "pending_tool_calls": [],
    }
    if stop_reason:
        update["status"] = STATUS_STOPPED
        update["stop_reason"] = stop_reason
    return update


def _plan_one(
    tool_call: dict[str, Any], *, state: HarnessState, ctx: HarnessContext
) -> dict[str, Any]:
    """Pha 1 cho một lời gọi: tra registry, validate tham số, hỏi policy.

    Không chạy gì, không ghi gì. Phải an toàn khi chạy lại nhiều lần (node sẽ chạy
    lại nguyên vẹn sau mỗi lần resume từ interrupt).
    """
    name = str(tool_call.get("name") or "")
    tool_call_id = str(tool_call.get("id") or f"call_{name}")
    raw_args = tool_call.get("args") or {}
    base = {"tool_call_id": tool_call_id, "name": name, "signature": ""}

    # Model gọi tool không tồn tại: chuyện thường, không phải lỗi hệ thống.
    if not ctx.registry.has(name):
        return {
            **base,
            "outcome": "failed",
            "failure": ToolFailure(
                f"không có tool nào tên {name!r}",
                code="unknown_tool",
                recoverable=True,
                hint=f"Các tool có thể dùng: {', '.join(ctx.registry.names())}",
            ),
        }

    spec: ToolSpec = ctx.registry.get(name)
    try:
        arguments = spec.validate_arguments(raw_args)
    except ToolFailure as exc:
        return {**base, "outcome": "failed", "failure": exc}

    signature = tool_call_signature(name, raw_args)
    decision = evaluate_tool_call(
        spec,
        raw_args,
        state=state,
        permissions=ctx.permissions,
        budget=ctx.budget,
    )

    if decision.decision == "deny":
        return {
            **base,
            "signature": signature,
            "outcome": "failed",
            "failure": PolicyDenied(decision.model_message or decision.reason, hint=decision.hint),
        }

    plan = {
        **base,
        "signature": signature,
        "spec": spec,
        "arguments": arguments,
        "outcome": "run",
    }
    if decision.needs_approval:
        plan["outcome"] = "needs_approval"
        plan["approval"] = PendingApproval(
            tool_call_id=tool_call_id,
            tool_name=name,
            arguments=raw_args,
            reason=decision.reason,
            risk=decision.risk,
        )
    return plan


def _normalize_decisions(
    raw: Any, requests: list[PendingApproval]
) -> dict[str, dict[str, Any]]:
    """Chuẩn hoá giá trị resume thành ``{tool_call_id: {approved, reason}}``.

    Nhận ba dạng, vì người tích hợp sẽ gửi dạng nào tiện nhất cho họ:
    ``True``/``False``, ``{"approved": bool, "reason": str}`` (áp cho tất cả), hoặc
    ``{"approvals": {id: {...}}}`` / ``{id: {...}}`` cho từng lời gọi.

    Mặc định khi không rõ là **từ chối**. Một giá trị resume méo mó không được trở
    thành "đồng ý" - đây là cổng duy nhất bảo vệ hành động ghi dữ liệu.
    """
    ids = [r["tool_call_id"] for r in requests]

    if raw is True:
        return {i: {"approved": True, "reason": "quyết định chung: đồng ý"} for i in ids}
    if raw is False:
        return {i: {"approved": False, "reason": "quyết định chung: từ chối"} for i in ids}

    if isinstance(raw, dict):
        inner = raw.get("approvals") if isinstance(raw.get("approvals"), dict) else None
        if inner is not None:
            return {
                i: _one_decision(inner.get(i))
                for i in ids
            }
        if "approved" in raw:
            shared = _one_decision(raw)
            return {i: shared for i in ids}
        # dict keyed theo tool_call_id
        if any(i in raw for i in ids):
            return {i: _one_decision(raw.get(i)) for i in ids}

    return {i: {"approved": False, "reason": "giá trị phê duyệt không hiểu được"} for i in ids}


def _one_decision(value: Any) -> dict[str, Any]:
    """Chuẩn hoá một quyết định. Chỉ ``True`` đúng nghĩa mới là đồng ý.

    Dùng ``is True`` chứ **không** dùng ``bool(...)``. Khác biệt này là một lỗ bảo
    mật thật đã bị test bắt: ``bool("có lẽ")`` là ``True``, nên một payload
    ``{"approved": "có lẽ"}`` - hay ``"no"``, hay ``"false"`` dạng chuỗi từ một form
    HTML - sẽ **mở cổng phê duyệt**. Với một cổng mà toàn bộ việc của nó là đứng
    giữa model và một hành động có hậu quả, "gần như là true" phải bị coi là false.
    """
    if value is True:
        return {"approved": True, "reason": ""}
    if isinstance(value, dict):
        approved = value.get("approved") is True
        reason = str(value.get("reason") or "")
        if not approved and value.get("approved") is not False:
            reason = reason or f"giá trị approved không phải boolean: {value.get('approved')!r}"
        return {"approved": approved, "reason": reason}
    return {"approved": False, "reason": "không có quyết định cho lời gọi này"}


# --------------------------------------------------------------------------- #
# finish
# --------------------------------------------------------------------------- #
def finish(state: HarnessState, runtime: Runtime[HarnessContext]) -> dict[str, Any]:
    """Chuẩn hoá kết thúc episode. Mọi đường đi đều qua đây."""
    status = state.get("status", STATUS_RUNNING)
    last_text = ""
    for message in reversed(state.get("messages") or []):
        if isinstance(message, AIMessage) and not message.tool_calls:
            last_text = str(message.content or "").strip()
            break

    if status == STATUS_STOPPED:
        answer = STOPPED_ANSWER_TEMPLATE.format(
            reason=state.get("stop_reason") or "không rõ",
            detail=_stop_detail(state),
            tool_calls=state.get("tool_calls_total", 0),
            model_calls=state.get("model_calls", 0),
        )
        if last_text:
            answer += f"\nPhần agent đã viết trước khi dừng:\n{last_text}"
        stop_reason = state.get("stop_reason")
    else:
        status = STATUS_ANSWERED
        stop_reason = STOP_ANSWERED
        answer = last_text or "(agent kết thúc mà không trả lời)"

    logger.info(
        "episode_finished",
        extra={
            "episode_id": state.get("episode_id"),
            "status": status,
            "stop_reason": stop_reason,
            "model_calls": state.get("model_calls"),
            "tool_calls_total": state.get("tool_calls_total"),
            "active_seconds": round(state.get("elapsed_active_seconds", 0.0), 3),
        },
    )
    return {
        "status": status,
        "stop_reason": stop_reason,
        "final_answer": answer,
        "pending_approval": None,
        "steps": [
            _step(state, "stop", stop_reason or "finished", "ok", 0.0,
                  {"answer_chars": len(answer)})
        ],
    }


def _stop_detail(state: HarnessState) -> str:
    for step in reversed(state.get("steps") or []):
        if step.get("kind") == "stop":
            return str((step.get("detail") or {}).get("message") or "")
    return ""


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def route_after_model(state: HarnessState) -> str:
    """Cạnh quyết định nên harness này là harness.

    Không có kế hoạch nào nói bước sau là gì. Câu trả lời phụ thuộc vào việc model
    vừa xin tool hay vừa trả lời - một dữ kiện chỉ có lúc chạy.
    """
    if state.get("status") != STATUS_RUNNING:
        return NODE_FINISH
    if state.get("pending_tool_calls"):
        return NODE_EXECUTE_TOOLS
    return NODE_FINISH


def route_after_tools(state: HarnessState) -> str:
    if state.get("status") != STATUS_RUNNING:
        return NODE_FINISH
    return NODE_CALL_MODEL


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #
def _step(
    state: HarnessState,
    kind: str,
    name: str,
    status: str,
    duration_ms: float,
    detail: dict[str, Any] | None = None,
) -> StepRecord:
    return StepRecord(
        index=len(state.get("steps") or []),
        kind=kind,
        name=name,
        status=status,
        duration_ms=round(duration_ms, 2),
        detail={k: v for k, v in (detail or {}).items() if v is not None},
    )


def _tool_error_message(tool_call_id: str, name: str, failure: ToolFailure) -> ToolMessage:
    """ToolMessage cho một lời gọi thất bại.

    ``status="error"`` để provider và model đều biết đây là lỗi. Nội dung viết cho
    model đọc và sửa, không phải cho người đọc log.
    """
    return ToolMessage(
        content=failure.to_tool_content(),
        tool_call_id=tool_call_id,
        name=name,
        status="error",
    )


# --------------------------------------------------------------------------- #
# Build graph
# --------------------------------------------------------------------------- #
def build_graph() -> StateGraph:
    builder: StateGraph = StateGraph(
        HarnessState,
        HarnessContext,
        input_schema=HarnessInput,
        output_schema=HarnessOutput,
    )
    builder.add_node(NODE_PREPARE, prepare)
    builder.add_node(NODE_CALL_MODEL, call_model)
    builder.add_node(NODE_EXECUTE_TOOLS, execute_tools)
    builder.add_node(NODE_FINISH, finish)

    builder.add_edge(START, NODE_PREPARE)
    builder.add_edge(NODE_PREPARE, NODE_CALL_MODEL)
    builder.add_conditional_edges(
        NODE_CALL_MODEL, route_after_model, [NODE_EXECUTE_TOOLS, NODE_FINISH]
    )
    builder.add_conditional_edges(
        NODE_EXECUTE_TOOLS, route_after_tools, [NODE_CALL_MODEL, NODE_FINISH]
    )
    builder.add_edge(NODE_FINISH, END)
    return builder


def build_harness(checkpointer: Any | None = None) -> Any:
    """Compile harness.

    ``recursion_limit`` của LangGraph là lớp chặn cuối cùng, nằm *dưới* các hạn mức
    trong ``BudgetPolicy``. Nếu nó là thứ chặn được một episode thì nghĩa là hạn mức
    phía trên đã bị cấu hình quá rộng - hãy sửa hạn mức, đừng dựa vào recursion_limit.
    """
    graph = build_graph().compile(checkpointer=checkpointer, name="agent_harness")
    logger.info("harness_compiled", extra={"nodes": len(graph.get_graph().nodes)})
    return graph
