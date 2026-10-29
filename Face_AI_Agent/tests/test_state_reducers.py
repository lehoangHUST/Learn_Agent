"""Test state schema và reducer.

Hai nhóm:

1. ``dedup_append`` - reducer tự viết: kiểm các tính chất mà một reducer *phải* có
   (thuần, chịu None, kết hợp được, idempotent với bản trùng).
2. Hành vi gộp thật trong một graph nhỏ, **bao gồm cả trường hợp hai node song song
   cùng ghi một field** - với reducer thì gộp đúng, không reducer thì
   ``InvalidUpdateError``. Đó là bằng chứng cho thấy reducer đang thật sự làm việc
   hợp nhất, chứ không phải "tình cờ đúng" nhờ chạy tuần tự.
"""

from __future__ import annotations

import operator
from typing import Annotated

import pytest
from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.agent.state import append_run_scoped, dedup_append


# --------------------------------------------------------------------------- #
# 1. Tính chất của reducer
# --------------------------------------------------------------------------- #
def test_dedup_append_concatenates() -> None:
    assert dedup_append(["a"], ["b"]) == ["a", "b"]


def test_dedup_append_removes_duplicates_keeping_order() -> None:
    assert dedup_append(["a", "b"], ["b", "c", "a", "d"]) == ["a", "b", "c", "d"]


def test_dedup_append_handles_none_current_value() -> None:
    """Lần gộp đầu tiên, channel chưa có giá trị -> reducer nhận ``current=None``."""
    assert dedup_append(None, ["a"]) == ["a"]
    assert dedup_append(None, None) == []


def test_update_none_means_reset_not_noop() -> None:
    """``update is None`` là tín hiệu "lượt mới, xoá sạch", không phải "không đổi".

    Cần một tín hiệu ngoài miền giá trị vì với phép nối thì ``[]`` không xoá được gì.
    Đây là cách state phạm vi-một-lần-chạy được dọn ở đầu mỗi lượt, khi checkpoint
    đã nạp lại state của lượt trước.
    """
    assert dedup_append(["a", "b"], None) == []
    assert append_run_scoped(["a", "b"], None) == []


def test_dedup_append_does_not_mutate_inputs() -> None:
    """Reducer phải thuần: nó được gọi lại khi resume từ checkpoint."""
    current = ["a"]
    update = ["b"]
    dedup_append(current, update)
    assert current == ["a"]
    assert update == ["b"]


def test_dedup_append_is_order_insensitive_as_a_set() -> None:
    """Hai update song song áp theo thứ tự nào cũng cho cùng *tập* phần tử.

    Thứ tự phần tử có thể khác, nên test khẳng định theo tập - đúng cách phải test
    một reducer dùng cho nhánh song song.
    """
    left, right = ["a", "b"], ["c", "b"]
    forward = dedup_append(dedup_append([], left), right)
    backward = dedup_append(dedup_append([], right), left)
    assert set(forward) == set(backward)
    assert len(forward) == len(backward) == 3


def test_dedup_append_is_idempotent() -> None:
    once = dedup_append(["a"], ["b"])
    twice = dedup_append(once, ["b"])
    assert once == twice


# --------------------------------------------------------------------------- #
# 2. Gộp thật trong graph
# --------------------------------------------------------------------------- #
class AccumulatingState(TypedDict):
    collected: Annotated[list[str], operator.add]
    warnings: Annotated[list[str], dedup_append]
    current_step: str  # reducer mặc định: ghi đè


def test_sequential_nodes_accumulate_and_overwrite() -> None:
    """Field tích lũy giữ tất cả; field ghi đè giữ giá trị của node chạy sau cùng."""
    builder: StateGraph = StateGraph(AccumulatingState)
    builder.add_node("first", lambda s: {"collected": ["a"], "current_step": "first"})
    builder.add_node("second", lambda s: {"collected": ["b"], "current_step": "second"})
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)

    out = builder.compile().invoke({"collected": [], "warnings": [], "current_step": ""})
    assert out["collected"] == ["a", "b"]
    assert out["current_step"] == "second"


def test_parallel_nodes_both_land_in_accumulating_field() -> None:
    """Hai nhánh song song cùng ghi field có reducer -> CẢ HAI update đều có mặt.

    Không khẳng định thứ tự: thứ tự giữa hai nhánh song song không được đảm bảo,
    nên một test khẳng định thứ tự sẽ pass hôm nay rồi flaky về sau.
    """
    builder: StateGraph = StateGraph(AccumulatingState)
    builder.add_node("left", lambda s: {"collected": ["from-left"]})
    builder.add_node("right", lambda s: {"collected": ["from-right"]})
    builder.add_node("join", lambda s: {"current_step": "joined"})
    builder.add_edge(START, "left")
    builder.add_edge(START, "right")
    builder.add_edge("left", "join")
    builder.add_edge("right", "join")
    builder.add_edge("join", END)

    out = builder.compile().invoke({"collected": [], "warnings": [], "current_step": ""})
    assert set(out["collected"]) == {"from-left", "from-right"}
    assert len(out["collected"]) == 2


def test_parallel_nodes_dedup_reducer_merges_without_duplicates() -> None:
    """Hai nhánh song song báo CÙNG một cảnh báo -> chỉ còn một bản."""
    same = "ngưỡng chưa hiệu chuẩn"
    builder: StateGraph = StateGraph(AccumulatingState)
    builder.add_node("left", lambda s: {"warnings": [same]})
    builder.add_node("right", lambda s: {"warnings": [same, "khác"]})
    builder.add_node("join", lambda s: {"current_step": "joined"})
    builder.add_edge(START, "left")
    builder.add_edge(START, "right")
    builder.add_edge("left", "join")
    builder.add_edge("right", "join")
    builder.add_edge("join", END)

    out = builder.compile().invoke({"collected": [], "warnings": [], "current_step": ""})
    assert sorted(out["warnings"]) == sorted([same, "khác"])


def test_parallel_write_to_overwrite_field_raises_invalid_update() -> None:
    """Hai node song song cùng ghi field KHÔNG có reducer -> InvalidUpdateError.

    LangGraph không chọn bừa một bên, vì thứ tự hoàn thành của hai nhánh là không
    xác định - chọn bừa sẽ cho một hệ thống không tất định và sai một cách im lặng.
    """
    builder: StateGraph = StateGraph(AccumulatingState)
    builder.add_node("left", lambda s: {"current_step": "left"})
    builder.add_node("right", lambda s: {"current_step": "right"})
    builder.add_edge(START, "left")
    builder.add_edge(START, "right")
    builder.add_edge("left", END)
    builder.add_edge("right", END)

    with pytest.raises(InvalidUpdateError):
        builder.compile().invoke({"collected": [], "warnings": [], "current_step": ""})


def test_sequential_write_to_same_overwrite_field_is_fine() -> None:
    """Đối chứng với test trên: cùng field, cùng hai node, nhưng TUẦN TỰ -> hợp lệ.

    Nói cách khác, InvalidUpdateError là lỗi về *đồng thời*, không phải về việc
    "hai node cùng ghi một field".
    """
    builder: StateGraph = StateGraph(AccumulatingState)
    builder.add_node("left", lambda s: {"current_step": "left"})
    builder.add_node("right", lambda s: {"current_step": "right"})
    builder.add_edge(START, "left")
    builder.add_edge("left", "right")
    builder.add_edge("right", END)

    out = builder.compile().invoke({"collected": [], "warnings": [], "current_step": ""})
    assert out["current_step"] == "right"


def test_omitted_key_leaves_channel_untouched() -> None:
    """Node không trả về key -> channel giữ nguyên, KHÁC với trả về None."""
    builder: StateGraph = StateGraph(AccumulatingState)
    builder.add_node("setter", lambda s: {"current_step": "set"})
    builder.add_node("silent", lambda s: {})  # không nhắc tới current_step
    builder.add_edge(START, "setter")
    builder.add_edge("setter", "silent")
    builder.add_edge("silent", END)

    out = builder.compile().invoke({"collected": [], "warnings": [], "current_step": ""})
    assert out["current_step"] == "set"


# --------------------------------------------------------------------------- #
# 3. Schema của agent thật
# --------------------------------------------------------------------------- #
def test_agent_state_has_no_binary_or_vector_fields() -> None:
    """State phải serialize được: nó đi vào checkpoint sau mỗi super-step.

    Test này khoá lại quyết định "không để ảnh/vector vào state" - ai thêm một field
    kiểu bytes/ndarray sẽ làm test này đỏ.
    """
    import numpy as np

    from app.agent.state import FaceAgentState

    hints = FaceAgentState.__annotations__
    forbidden = (bytes, bytearray, memoryview, np.ndarray)
    for name, annotation in hints.items():
        text = str(annotation)
        for bad in forbidden:
            assert bad.__name__ not in text, f"field {name!r} mang kiểu không serialize được"


# --------------------------------------------------------------------------- #
# 4. Dữ liệu tổng hợp phải có cấu trúc đúng, nếu không mọi test dựa trên nó vô nghĩa
# --------------------------------------------------------------------------- #
def test_identity_vector_same_identity_different_variant_is_same_person() -> None:
    """Hai "ảnh" của cùng một người -> cosine rất cao, và hai vector khác nhau.

    Đây là bất biến mà toàn bộ test matching dựa vào. Nếu nó sai (ví dụ đổi chuỗi
    identity để tạo ảnh thứ hai), tập "cùng người" sẽ có phân bố y như tập "khác
    người" và mọi test ngưỡng vẫn xanh trong khi không kiểm chứng điều gì.
    """
    import numpy as np

    from app.vision.fake import identity_vector

    base = identity_vector("person-A", 512)
    photo2 = identity_vector("person-A", 512, jitter=0.1, variant=1)
    photo3 = identity_vector("person-A", 512, jitter=0.1, variant=2)

    assert float(base @ photo2) > 0.98
    assert float(base @ photo3) > 0.98
    assert not np.allclose(photo2, photo3), "hai ảnh khác nhau phải cho vector khác nhau"
    assert float(np.linalg.norm(base)) == pytest.approx(1.0, abs=1e-5)


def test_identity_vector_different_identity_is_different_person() -> None:
    from app.vision.fake import identity_vector

    assert abs(float(identity_vector("person-A", 512) @ identity_vector("person-B", 512))) < 0.2


def test_changing_identity_string_creates_a_different_person() -> None:
    """Khoá lại cái bẫy: hậu tố trong identity KHÔNG phải là "ảnh khác của cùng người"."""
    from app.vision.fake import identity_vector

    assert abs(float(identity_vector("person-A", 512) @ identity_vector("person-A#1", 512))) < 0.2
