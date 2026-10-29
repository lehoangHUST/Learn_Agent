"""Test vòng lặp harness - những hành vi mà model thật không chịu tạo ra khi cần.

Mỗi test ở đây dựng chính xác một chế độ thất bại của agent tool-calling và khẳng
định harness xử lý đúng. Đó là lý do tồn tại của ``ScriptedModelClient``: không thể
viết test tin cậy cho "model lặp vô hạn" hay "model gọi tool không tồn tại" bằng
một model thật.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from harness.budget import BudgetPolicy
from harness.model import call, calls, react_to_last_tool_result, say
from harness.state import (
    STATUS_ANSWERED,
    STATUS_STOPPED,
    STOP_MAX_MODEL_CALLS,
    STOP_MAX_TOOL_CALLS,
)
from tests.conftest import USER, assert_every_tool_call_answered, tool_steps


# --------------------------------------------------------------------------- #
# Đường đi lành mạnh
# --------------------------------------------------------------------------- #
def test_model_answers_without_tools(make_runner) -> None:
    """Một lượt không có tool call là cách kết thúc episode."""
    runner = make_runner([say("Tôi cần biết bạn muốn phân tích gì.")])
    result = runner.start("xin chào", user_id=USER)

    assert result.status == STATUS_ANSWERED
    assert result.output["model_calls"] == 1
    assert result.output["tool_calls_total"] == 0
    assert "phân tích gì" in result.final_answer


def test_tool_then_answer(make_runner) -> None:
    runner = make_runner(
        [
            calls(call("list_tables")),
            say("Database có các bảng customers, orders, products, order_items."),
        ]
    )
    result = runner.start("database có gì?", user_id=USER)

    assert result.status == STATUS_ANSWERED
    assert result.output["tool_calls_total"] == 1
    assert result.output["tool_calls_by_name"] == {"list_tables": 1}
    assert_every_tool_call_answered(runner, result.thread_id)


def test_tool_result_reaches_the_model(make_runner) -> None:
    """Kết quả tool phải thực sự được đưa vào lượt gọi model kế tiếp.

    Khẳng định bằng cách để kịch bản *đọc* nội dung ToolMessage - nếu harness không
    truyền kết quả vào, nhánh rẽ sẽ khác và test đỏ.
    """
    runner = make_runner(
        [
            calls(call("list_tables")),
            react_to_last_tool_result(
                lambda text: "customers" in text,
                say("Thấy bảng customers trong kết quả."),
                say("Không thấy gì."),
            ),
        ]
    )
    result = runner.start("liệt kê bảng", user_id=USER)
    assert "Thấy bảng customers" in result.final_answer


def test_parallel_tool_calls_in_one_turn(make_runner) -> None:
    runner = make_runner(
        [
            calls(
                call("list_tables", call_id="a"),
                call("describe_table", call_id="b", table="customers"),
            ),
            say("xong"),
        ]
    )
    result = runner.start("schema?", user_id=USER)

    assert result.output["tool_calls_total"] == 2
    assert result.output["model_calls"] == 2  # hai tool trong MỘT lượt
    assert_every_tool_call_answered(runner, result.thread_id)


# --------------------------------------------------------------------------- #
# Model làm sai - harness phải tha thứ và báo lại
# --------------------------------------------------------------------------- #
def test_unknown_tool_becomes_a_tool_message_not_a_crash(make_runner) -> None:
    runner = make_runner(
        [
            calls(call("drop_everything", confirm=True)),
            react_to_last_tool_result(
                lambda text: "unknown_tool" in text and "list_tables" in text,
                say("Tool đó không có; gợi ý đã liệt kê tool hợp lệ."),
                say("Không nhận được gợi ý."),
            ),
        ]
    )
    result = runner.start("xoá hết đi", user_id=USER)

    assert result.status == STATUS_ANSWERED  # episode không sập
    assert "gợi ý đã liệt kê tool hợp lệ" in result.final_answer
    # Lời gọi không chạy -> không tính vào hạn mức tool.
    assert result.output["tool_calls_total"] == 0
    assert tool_steps(result.output)[0]["detail"]["code"] == "unknown_tool"
    assert_every_tool_call_answered(runner, result.thread_id)


def test_invalid_arguments_are_reported_with_schema(make_runner) -> None:
    """Tham số sai schema -> lỗi kèm schema để model sửa được."""
    runner = make_runner(
        [
            calls(call("run_query", sql=123, purpose="x")),  # sai kiểu + purpose quá ngắn
            react_to_last_tool_result(
                lambda text: "invalid_arguments" in text and "properties" in text,
                say("Đã nhận schema và sẽ sửa."),
                say("Không nhận được schema."),
            ),
        ]
    )
    result = runner.start("đếm khách hàng", user_id=USER)
    assert "Đã nhận schema" in result.final_answer


def test_model_recovers_from_sql_error(make_runner) -> None:
    """Luồng tự sửa: cột sai -> đọc lỗi -> xem schema -> truy vấn đúng."""
    runner = make_runner(
        [
            calls(
                call(
                    "run_query",
                    sql="SELECT total_spend FROM customers",
                    purpose="Tổng chi tiêu",
                )
            ),
            react_to_last_tool_result(
                lambda text: "unknown_object" in text or "no such column" in text,
                calls(call("describe_table", table="customers")),
                say("không hiểu lỗi"),
            ),
            calls(
                call(
                    "run_query",
                    sql="SELECT COUNT(*) AS n FROM customers",
                    purpose="Đếm khách hàng",
                )
            ),
            say("Đã sửa truy vấn sau khi xem schema."),
        ]
    )
    result = runner.start("khách hàng chi bao nhiêu?", user_id=USER)

    assert result.status == STATUS_ANSWERED
    assert len(tool_steps(result.output, "run_query")) == 2
    assert tool_steps(result.output, "run_query")[0]["status"] == "error"
    assert tool_steps(result.output, "run_query")[1]["status"] == "ok"


def test_tool_crash_is_contained(make_runner, settings) -> None:
    """Handler raise exception lạ -> thành ToolMessage lỗi, episode vẫn đi tiếp."""
    from pydantic import BaseModel

    from harness.registry import ToolRegistry, ToolSpec

    class NoArgs(BaseModel):
        model_config = {"extra": "forbid"}

    def exploding(args: NoArgs, runtime: Any) -> str:
        raise ZeroDivisionError("handler của domain hỏng")

    registry = ToolRegistry(
        [
            ToolSpec(
                name="explode",
                description="tool luôn lỗi",
                args_model=NoArgs,
                handler=exploding,
            )
        ]
    )
    runner = make_runner(
        [
            calls(call("explode")),
            react_to_last_tool_result(
                lambda text: "tool_crashed" in text,
                say("Tool lỗi nhưng tôi vẫn trả lời được."),
                say("Không thấy lỗi."),
            ),
        ],
        registry=registry,
    )
    result = runner.start("chạy explode", user_id=USER)

    assert result.status == STATUS_ANSWERED
    assert "vẫn trả lời được" in result.final_answer
    assert tool_steps(result.output, "explode")[0]["detail"]["code"] == "tool_crashed"


# --------------------------------------------------------------------------- #
# Hạn mức - thứ làm nên an toàn của harness
# --------------------------------------------------------------------------- #
def test_identical_calls_are_denied_before_budget_runs_out(make_runner) -> None:
    """Lời gọi lặp y nguyên bị từ chối, kèm lời khuyên đổi cách - không dừng episode."""
    repeated = calls(
        call("run_query", sql="SELECT COUNT(*) FROM orders", purpose="Đếm đơn")
    )
    runner = make_runner([repeated, repeated, repeated, say("dừng lặp")])
    result = runner.start("bao nhiêu đơn?", user_id=USER)

    steps = tool_steps(result.output, "run_query")
    # max_identical_tool_calls=2 -> hai lần đầu chạy, lần thứ ba bị chặn.
    assert [s["status"] for s in steps] == ["ok", "ok", "error"]
    assert steps[2]["detail"]["code"] == "policy_denied"
    assert result.output["tool_calls_total"] == 2
    assert result.status == STATUS_ANSWERED  # vẫn kết thúc bình thường


def test_runaway_loop_is_stopped_by_model_call_budget(make_runner) -> None:
    """Model không bao giờ dừng -> harness dừng, với trạng thái và lý do rõ ràng."""
    repeated = calls(call("list_tables"))
    runner = make_runner([repeated] * 20)
    result = runner.start("lặp mãi", user_id=USER)

    assert result.status == STATUS_STOPPED
    assert result.output["stop_reason"] == STOP_MAX_MODEL_CALLS
    assert result.output["model_calls"] == 6  # đúng hạn mức, không hơn
    # Hết hạn mức KHÔNG phải exception: người gọi vẫn nhận câu trả lời dùng được.
    assert "Harness đã dừng" in result.final_answer
    assert STOP_MAX_MODEL_CALLS in result.final_answer


def test_per_tool_quota_denies_without_stopping_episode(make_runner) -> None:
    """Hết quota của MỘT tool thì model vẫn dùng được tool khác."""
    runner = make_runner(
        [
            calls(call("list_tables", call_id="t1")),
            calls(call("list_tables", call_id="t2")),
            calls(call("list_tables", call_id="t3")),  # quota list_tables = 2
            calls(call("describe_table", table="orders")),
            say("Đã chuyển sang tool khác."),
        ]
    )
    result = runner.start("schema", user_id=USER)

    assert result.status == STATUS_ANSWERED
    assert result.output["tool_calls_by_name"]["list_tables"] == 2
    assert result.output["tool_calls_by_name"]["describe_table"] == 1
    assert tool_steps(result.output, "list_tables")[2]["status"] == "error"


def test_tool_call_budget_stops_episode(make_runner, settings) -> None:
    """Vượt hạn mức tool tổng -> dừng episode (khác với hết quota một tool)."""
    runner = make_runner(
        [
            calls(
                *(
                    call("describe_table", call_id=f"{prefix}{i}", table=table)
                    for i, table in enumerate(["customers", "orders", "products"])
                )
            )
            for prefix in ("d", "e", "f")
        ]
        + [say("xong")],
        # Tắt chặn-lặp cho test này: ba lượt gọi describe_table với cùng tham số là
        # lời gọi lặp, và chặn-lặp sẽ kích hoạt TRƯỚC hạn mức tổng. Test này nói về
        # hạn mức tổng, nên phải cô lập đúng cơ chế đó.
        override={"budget": dataclasses.replace(settings.budget, max_identical_tool_calls=99)},
    )
    result = runner.start("mô tả hết", user_id=USER)

    assert result.output["stop_reason"] == STOP_MAX_TOOL_CALLS
    assert result.status == STATUS_STOPPED
    assert result.output["tool_calls_total"] == 8  # max_tool_calls


def test_parallel_cap_still_answers_every_tool_call_id(make_runner) -> None:
    """Lời gọi bị hoãn vẫn phải có ToolMessage - nếu không provider sẽ từ chối lượt sau."""
    runner = make_runner(
        [
            calls(
                call("list_tables", call_id="p1"),
                call("describe_table", call_id="p2", table="customers"),
                call("describe_table", call_id="p3", table="orders"),
                call("describe_table", call_id="p4", table="products"),
                call("describe_table", call_id="p5", table="order_items"),
            ),
            say("xong"),
        ]
    )
    result = runner.start("mô tả hết", user_id=USER)

    # max_parallel_tool_calls=3 -> 3 chạy, 2 bị hoãn.
    assert result.output["tool_calls_total"] == 3
    skipped = [s for s in tool_steps(result.output) if s["status"] == "skipped"]
    assert len(skipped) == 2
    assert all(s["detail"]["reason"] == "parallel_limit" for s in skipped)
    assert_every_tool_call_answered(runner, result.thread_id)


def test_row_limit_warns_the_model_data_is_partial(make_runner) -> None:
    """Lớp cắt thứ nhất: giới hạn số DÒNG ở tầng SQL.

    ``SELECT * FROM order_items`` trên 400 dòng nhưng ``max_rows=20`` -> kết quả chỉ
    có 20 dòng, và phải nói rõ là bị cắt. Nếu không nói, model sẽ kết luận "bảng có
    20 dòng".
    """
    runner = make_runner(
        [
            calls(call("run_query", sql="SELECT * FROM order_items", purpose="Xem dòng đơn")),
            react_to_last_tool_result(
                lambda text: "KHÔNG phải toàn bộ dữ liệu" in text,
                say("Kết quả chỉ là một phần nên tôi sẽ dùng COUNT(*)."),
                say("Dữ liệu đầy đủ."),
            ),
        ]
    )
    result = runner.start("xem order_items", user_id=USER)
    assert "chỉ là một phần" in result.final_answer


def test_harness_truncates_oversized_tool_result(make_runner, settings) -> None:
    """Lớp cắt thứ hai: giới hạn KÝ TỰ ở tầng harness.

    Hai lớp khác nhau và cần cả hai: giới hạn dòng không chặn được một cột TEXT dài,
    còn giới hạn ký tự là thứ bảo vệ context window.
    """
    runner = make_runner(
        [
            calls(call("run_query", sql="SELECT * FROM order_items", purpose="Xem dòng đơn")),
            react_to_last_tool_result(
                lambda text: "harness" in text and "bị cắt" in text,
                say("Harness đã cắt kết quả nên tôi không kết luận trên phần nhìn thấy."),
                say("Dữ liệu đầy đủ."),
            ),
        ],
        override={"budget": dataclasses.replace(settings.budget, max_tool_result_chars=400)},
    )
    result = runner.start("xem order_items", user_id=USER)

    step = tool_steps(result.output, "run_query")[0]
    assert step["detail"]["truncated_chars"] > 0
    assert step["detail"]["result_chars"] <= 400 + 200  # hạn mức + dòng thông báo
    assert "không kết luận trên phần nhìn thấy" in result.final_answer


def test_total_result_budget_stops_episode(make_runner) -> None:
    """Nhiều kết quả vừa phải cộng lại cũng làm nổ context -> có hạn mức tổng."""
    runner = make_runner(
        [
            calls(call("run_query", call_id=f"q{i}", sql="SELECT * FROM order_items", purpose=f"lần {i}"))
            for i in range(6)
        ]
        + [say("xong")],
        override={
            "budget": BudgetPolicy(
                max_model_calls=10,
                max_tool_calls=20,
                max_tool_result_chars=1500,
                max_total_result_chars=3000,
                max_identical_tool_calls=99,
            )
        },
    )
    result = runner.start("kéo nhiều dữ liệu", user_id=USER)

    assert result.status == STATUS_STOPPED
    assert result.output["stop_reason"] == "budget_result_chars"


# --------------------------------------------------------------------------- #
# Tính tất định
# --------------------------------------------------------------------------- #
def test_same_script_gives_same_transcript(make_runner) -> None:
    """Cùng kịch bản -> cùng transcript. Là nền để so sánh hai lần chạy khi debug."""
    script = [calls(call("list_tables")), say("xong")]

    def signature(output: dict[str, Any]) -> list[tuple[Any, ...]]:
        return [(s["kind"], s["name"], s["status"]) for s in output["steps"]]

    first = make_runner(script).start("x", user_id=USER)
    second = make_runner(script).start("x", user_id=USER)

    assert signature(first.output) == signature(second.output)
    assert first.output["tool_calls_by_name"] == second.output["tool_calls_by_name"]


def test_scripted_model_sees_the_tool_catalog(make_runner, settings) -> None:
    """Model phải được đưa đúng tập tool - khẳng định tường minh, không giả định."""
    from harness.model import ScriptedModelClient

    model = ScriptedModelClient([say("xong")])
    runner = make_runner([], override=None)
    # Dựng lại runner với model này để quan sát invocations.
    from langgraph.checkpoint.memory import InMemorySaver

    from app.wiring import build_runner

    runner = build_runner(settings, model=model, checkpointer=InMemorySaver())
    runner.start("x", user_id=USER)

    offered = model.invocations[0].tool_names
    assert set(offered) == {
        "list_tables",
        "describe_table",
        "run_query",
        "profile_column",
        "save_finding",
    }
    # Lượt đầu: system + human.
    assert model.invocations[0].message_count == 2
    assert model.invocations[0].last_message_type == "human"


def test_empty_goal_is_a_harness_error(make_runner) -> None:
    """Goal rỗng là lỗi của người gọi, không phải của model -> raise ra ngoài."""
    from harness.errors import HarnessError

    runner = make_runner([say("x")])
    with pytest.raises(HarnessError):
        runner.start("   ", user_id=USER)
