"""CLI: chạy một episode và in transcript.

::

    python -m app.cli --demo list                       # liệt kê kịch bản demo
    python -m app.cli --demo injection --approve deny    # chạy offline, không cần API key
    python -m app.cli --goal "Doanh thu theo tháng?"     # cần OPENAI_API_KEY

Phần phê duyệt mặc định là **hỏi trên stdin**. Đó không phải chi tiết trình bày:
một harness có tool ghi dữ liệu thì phải có một con người ở đâu đó trong vòng lặp,
và CLI là chỗ con người đó xuất hiện. ``--approve approve|deny`` dùng cho chạy
không tương tác (CI, demo).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_settings
from app.demo_scripts import GOALS, SCENARIOS
from app.wiring import build_runner
from harness.errors import HarnessError, ModelNotConfigured
from harness.runner import EpisodeResult, HarnessRunner
from harness.transcript import render_transcript, summarize


def configure_logging(level: str, *, quiet: bool) -> None:
    logging.basicConfig(
        level="WARNING" if quiet else level.upper(),
        format="%(levelname)-7s %(name)s | %(message)s",
    )


def prompt_for_approval(result: EpisodeResult, mode: str) -> dict[str, dict[str, object]]:
    """Thu quyết định phê duyệt cho mọi lời gọi đang chờ.

    Trả về dict theo ``tool_call_id`` thay vì một quyết định chung: một lượt có thể
    xin hai hành động ghi và người phê duyệt có quyền đồng ý cái này, từ chối cái kia.
    """
    decisions: dict[str, dict[str, object]] = {}
    for request in result.pending:
        print()
        print("-" * 72)
        print("CẦN PHÊ DUYỆT")
        print(f"  tool    : {request.get('tool_name')}")
        print(f"  rủi ro  : {request.get('risk')}")
        print("  tham số :")
        print(json.dumps(request.get("arguments"), ensure_ascii=False, indent=4))
        print("-" * 72)

        if mode in ("approve", "deny"):
            approved = mode == "approve"
            print(f"  -> tự động {'ĐỒNG Ý' if approved else 'TỪ CHỐI'} (--approve {mode})")
            reason = f"quyết định tự động: {mode}"
        else:
            try:
                answer = input("  Đồng ý chạy? [y/N]: ").strip().lower()
            except EOFError:
                # Không có stdin (chạy trong pipe/CI) mà cũng không đặt --approve:
                # mặc định TỪ CHỐI. Không bao giờ coi "không trả lời" là đồng ý.
                answer = ""
                print("  (không đọc được stdin -> TỪ CHỐI)")
            approved = answer in ("y", "yes", "đ", "co", "có")
            reason = "người dùng đồng ý qua CLI" if approved else "người dùng từ chối qua CLI"

        decisions[str(request.get("tool_call_id"))] = {"approved": approved, "reason": reason}
    return decisions


def run_episode(runner: HarnessRunner, goal: str, *, user_id: str, approve_mode: str) -> int:
    result = runner.start(goal, user_id=user_id)

    # Một episode có thể dừng chờ phê duyệt nhiều lần (nhiều lượt có tool ghi).
    rounds = 0
    while result.awaiting_approval:
        rounds += 1
        if rounds > 10:  # pragma: no cover - chặn vòng lặp phê duyệt bất thường
            print("Quá nhiều vòng phê duyệt, dừng.")
            break
        decisions = prompt_for_approval(result, approve_mode)
        result = runner.resume(result.thread_id, decisions, user_id=user_id)

    print()
    print(render_transcript(result.output))
    print()
    print("TÓM TẮT:", json.dumps(summarize(result.output), ensure_ascii=False))
    return 0 if result.status in ("answered", "stopped") else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--goal", help="câu hỏi cho agent (dùng model thật)")
    source.add_argument("--demo", help="tên kịch bản demo, hoặc 'list'")
    parser.add_argument("--user", default="cli-user")
    parser.add_argument(
        "--approve",
        choices=["ask", "approve", "deny"],
        default="ask",
        help="cách xử lý yêu cầu phê duyệt (mặc định: hỏi trên stdin)",
    )
    parser.add_argument("--quiet", action="store_true", help="chỉ in transcript, bỏ log")
    args = parser.parse_args()

    settings = load_settings()
    configure_logging(settings.log_level, quiet=args.quiet)

    if args.demo == "list":
        print("Các kịch bản demo (chạy được offline, không cần API key):")
        width = max(len(name) for name in SCENARIOS)
        for name, (description, _) in SCENARIOS.items():
            print(f"  {name:<{width}}  {description}")
        print()
        print("Ví dụ: python -m app.cli --demo injection --approve deny --quiet")
        return 0

    if not settings.db_path.exists():
        print(f"Không tìm thấy database tại {settings.db_path}")
        print("Chạy: python -m toolsets.sql.seed")
        return 1

    if args.demo:
        if args.demo not in SCENARIOS:
            print(f"Không có kịch bản {args.demo!r}. Dùng --demo list để xem danh sách.")
            return 1
        description, factory = SCENARIOS[args.demo]
        goal = GOALS.get(args.demo, "Phân tích dữ liệu")
        print(f"KỊCH BẢN: {args.demo} - {description}")
        print(f"CÂU HỎI : {goal}")
        runner = build_runner(settings, script=factory())
    else:
        goal = args.goal
        try:
            runner = build_runner(settings)
        except ModelNotConfigured as exc:
            print(f"Không dựng được model: {exc}")
            print("Đặt OPENAI_API_KEY, hoặc dùng --demo <tên> để chạy offline.")
            return 1

    try:
        return run_episode(runner, goal, user_id=args.user, approve_mode=args.approve)
    except HarnessError as exc:
        # HarnessError là lỗi cấu hình/bất biến, khác với lỗi của model.
        print(f"LỖI HARNESS [{exc.code}]: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
