"""Hiển thị transcript của một episode.

Vì sao transcript là một thành phần chứ không phải một dòng log: trong một harness,
**không ai biết trước agent đã làm gì**. Với một workflow thì đọc code là biết đường
đi; với harness thì đường đi do model chọn lúc chạy, nên câu hỏi "nó đã làm gì để
ra kết luận này" chỉ trả lời được bằng bản ghi.

Transcript ở đây trả lời đúng bốn câu hỏi, và đó cũng là bốn câu hay phải trả lời
khi một episode cho kết quả lạ:

1. Model đã xin tool nào, theo thứ tự nào?
2. Lời gọi nào thất bại, vì sao, và model có sửa được không?
3. Episode đã tiêu bao nhiêu hạn mức?
4. Có hành động nào cần phê duyệt, ai đã đồng ý?
"""

from __future__ import annotations

from typing import Any

_STATUS_MARK = {
    "ok": "+",
    "error": "!",
    "denied": "x",
    "skipped": "-",
}


def render_transcript(output: dict[str, Any], *, verbose: bool = False) -> str:
    """Bản text của một episode, đọc được bằng mắt."""
    lines: list[str] = []
    status = output.get("status", "?")
    lines.append("=" * 72)
    lines.append(f"EPISODE {output.get('episode_id', '?')}  [{status}]")
    lines.append(f"thread: {output.get('thread_id', '?')}")
    if output.get("stop_reason"):
        lines.append(f"kết thúc vì: {output['stop_reason']}")
    lines.append("=" * 72)

    lines.append("")
    lines.append("CÁC BƯỚC")
    for step in output.get("steps") or []:
        kind = step.get("kind")
        if kind == "model" and step.get("name") == "prepare":
            detail = step.get("detail") or {}
            lines.append(f"  .  prepare            tool có sẵn: {', '.join(detail.get('tools_offered', []))}")
            continue

        mark = _STATUS_MARK.get(str(step.get("status")), "?")
        name = str(step.get("name"))
        duration = step.get("duration_ms", 0.0)
        detail = step.get("detail") or {}

        if kind == "model":
            requested = detail.get("tool_calls_requested") or []
            what = f"xin tool: {', '.join(requested)}" if requested else "trả lời bằng văn bản"
            lines.append(f"  {mark}  model  {name:<18} {what}  ({duration:.0f}ms)")
        elif kind == "tool":
            bits = []
            if detail.get("result_chars"):
                bits.append(f"{detail['result_chars']} ký tự")
            if detail.get("truncated_chars"):
                bits.append(f"cắt {detail['truncated_chars']}")
            if detail.get("code"):
                bits.append(str(detail["code"]))
            if detail.get("reason"):
                bits.append(str(detail["reason"]))
            suffix = f"  [{', '.join(bits)}]" if bits else ""
            lines.append(f"  {mark}  tool   {name:<18}{suffix}  ({duration:.0f}ms)")
        elif kind == "approval":
            lines.append(f"  {mark}  phê duyệt {name:<15} bởi {detail.get('approved_by', '?')}")
        elif kind == "stop":
            message = detail.get("message") or ""
            lines.append(f"  {mark}  kết thúc {name}{(': ' + str(message)) if message else ''}")

    lines.append("")
    lines.append("HẠN MỨC ĐÃ DÙNG")
    lines.append(f"  lượt gọi model : {output.get('model_calls', 0)}")
    lines.append(f"  lượt gọi tool  : {output.get('tool_calls_total', 0)}")
    by_name = output.get("tool_calls_by_name") or {}
    if by_name:
        inner = ", ".join(f"{k}={v}" for k, v in sorted(by_name.items()))
        lines.append(f"  theo tool      : {inner}")
    lines.append(f"  thời gian làm việc: {output.get('elapsed_active_seconds', 0.0):.2f}s")

    pending = output.get("pending_approval")
    if pending:
        lines.append("")
        lines.append("ĐANG CHỜ PHÊ DUYỆT")
        lines.append(f"  tool: {pending.get('tool_name')}")
        lines.append(f"  rủi ro: {pending.get('risk')}")

    lines.append("")
    lines.append("CÂU TRẢ LỜI")
    lines.append(output.get("final_answer") or "(không có)")

    if verbose:
        lines.append("")
        lines.append("CHI TIẾT TỪNG BƯỚC")
        for step in output.get("steps") or []:
            lines.append(f"  {step.get('index')}: {step}")

    return "\n".join(lines)


def summarize(output: dict[str, Any]) -> dict[str, Any]:
    """Bản gọn để log hoặc trả qua API."""
    tool_steps = [s for s in (output.get("steps") or []) if s.get("kind") == "tool"]
    return {
        "episode_id": output.get("episode_id"),
        "thread_id": output.get("thread_id"),
        "status": output.get("status"),
        "stop_reason": output.get("stop_reason"),
        "model_calls": output.get("model_calls", 0),
        "tool_calls_total": output.get("tool_calls_total", 0),
        "tool_calls_by_name": output.get("tool_calls_by_name") or {},
        "tool_failures": sum(1 for s in tool_steps if s.get("status") == "error"),
        "tool_denied": sum(1 for s in tool_steps if s.get("status") == "denied"),
        "elapsed_active_seconds": round(output.get("elapsed_active_seconds", 0.0), 3),
        "answer_chars": len(output.get("final_answer") or ""),
    }
