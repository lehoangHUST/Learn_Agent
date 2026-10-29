"""Dựng và compile graph.

Hình dạng graph (``python -m scripts.draw_graph`` in ra Mermaid thật từ chính object
này, nên sơ đồ không bao giờ lệch với code)::

    START
      -> begin_run            (dọn state của lượt trước mà checkpoint nạp lại)
      -> validate_input
      -> classify_intent
      -> plan_execution
      -> [detect_faces] -> [generate_embeddings] -> [search_faces]
      -> analyze_results
      -> answer_question
      -> finalize_response -> END

    Mọi node đều có thể rẽ sang handle_error; handle_error rẽ về node cần retry,
    hoặc sang answer_question để giải thích, hoặc sang finalize_response.

Các bước trong ngoặc vuông chỉ chạy nếu ``execution_plan`` có chúng - xem
``app/agent/plan.py``.

Hai quyết định đáng giải thích
-----------------------------
**1. Vì sao không dùng ``retry_policy`` của LangGraph.**
``RetryPolicy`` chỉ hoạt động khi exception *thoát ra khỏi* node. Nhưng decorator
``graph_node`` cố tình bắt mọi exception và biến thành ``ErrorEntry`` trong state,
vì người gọi API cần một response đúng schema kèm ``request_id`` chứ không phải một
stack trace. Hai cơ chế đó loại trừ nhau; nếu khai báo cả hai thì ``retry_policy``
thành code chết và gây hiểu sai cho người đọc sau.

Nên retry nằm ở đúng hai tầng, cả hai đều thật:

* **Tầng transport** - ``ImintV3FaceEngine._post_with_retry``: retry timeout và 5xx
  với backoff luỹ thừa, không retry 4xx.
* **Tầng luồng** - ``handle_error`` -> cạnh quay lại node đã lỗi, có đếm
  ``retry_count`` và chặn bởi ``graph.max_node_retries``.

**2. Vì sao graph tuần tự chứ không fan-out theo từng khuôn mặt.**
Một ảnh nhiều mặt nghe như bài toán map-reduce, nhưng FAISS tìm theo batch trong
một lời gọi (xem ``BaseFaceSearchRepository.search`` nhận ma trận). Fan-out n nhánh
để mỗi nhánh làm một phần của việc vốn đã là một lời gọi thì chỉ thêm n super-step
và thêm nhu cầu reducer, mà không nhanh hơn. Khi nào fan-out mới đáng: nhiều ảnh
*và* backend vision chịu được request song song - lúc đó ``Send`` cho từng ảnh là
đúng chỗ.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agent import routing
from app.agent.nodes import (
    analyze_results,
    answer_question,
    begin_run,
    classify_intent,
    detect_faces,
    finalize_response,
    generate_embeddings,
    handle_error,
    plan_execution,
    search_faces,
    validate_input,
)
from app.agent.state import (
    AnswerNodeInput,
    FaceAgentContext,
    FaceAgentInput,
    FaceAgentOutput,
    FaceAgentState,
)

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """Dựng builder (chưa compile) - tách riêng để test có thể kiểm tra topology."""
    builder: StateGraph = StateGraph(
        FaceAgentState,
        FaceAgentContext,
        input_schema=FaceAgentInput,
        output_schema=FaceAgentOutput,
    )

    # begin_run không có decorator graph_node - xem docstring của nó.
    builder.add_node(routing.NODE_BEGIN, begin_run)
    builder.add_node(routing.NODE_VALIDATE, validate_input)
    builder.add_node(routing.NODE_INTENT, classify_intent)
    builder.add_node(routing.NODE_PLAN, plan_execution)
    builder.add_node(routing.NODE_DETECT, detect_faces)
    builder.add_node(routing.NODE_EMBED, generate_embeddings)
    builder.add_node(routing.NODE_SEARCH, search_faces)
    builder.add_node(routing.NODE_ANALYZE, analyze_results)
    # input_schema hẹp: node duy nhất gọi LLM chỉ nhìn thấy phần dữ liệu nó cần.
    builder.add_node(routing.NODE_ANSWER, answer_question, input_schema=AnswerNodeInput)
    builder.add_node(routing.NODE_ERROR, handle_error)
    builder.add_node(routing.NODE_FINALIZE, finalize_response)

    builder.add_edge(START, routing.NODE_BEGIN)
    builder.add_edge(routing.NODE_BEGIN, routing.NODE_VALIDATE)

    builder.add_conditional_edges(
        routing.NODE_VALIDATE, routing.route_after_validate, routing.PATHS_AFTER_VALIDATE
    )
    builder.add_conditional_edges(
        routing.NODE_INTENT, routing.route_after_intent, routing.PATHS_AFTER_INTENT
    )
    builder.add_conditional_edges(
        routing.NODE_PLAN, routing.route_after_plan, routing.PATHS_AFTER_PLAN
    )
    builder.add_conditional_edges(
        routing.NODE_DETECT, routing.route_after_detect, routing.PATHS_AFTER_DETECT
    )
    builder.add_conditional_edges(
        routing.NODE_EMBED, routing.route_after_embed, routing.PATHS_AFTER_EMBED
    )
    builder.add_conditional_edges(
        routing.NODE_SEARCH, routing.route_after_search, routing.PATHS_AFTER_SEARCH
    )
    builder.add_conditional_edges(
        routing.NODE_ANALYZE, routing.route_after_analyze, routing.PATHS_AFTER_ANALYZE
    )
    builder.add_conditional_edges(
        routing.NODE_ANSWER, routing.route_after_answer, [routing.NODE_FINALIZE]
    )
    builder.add_conditional_edges(
        routing.NODE_ERROR, routing.route_after_error, routing.PATHS_AFTER_ERROR
    )

    builder.add_edge(routing.NODE_FINALIZE, END)
    return builder


def build_agent_graph(checkpointer: Any | None = None) -> Any:
    """Compile graph.

    ``compile()`` kiểm tra topology (node không tới được, cạnh trỏ vào node không
    tồn tại) nên lỗi cấu trúc nổ ra lúc khởi động, không phải giữa một request.
    """
    graph = build_graph().compile(checkpointer=checkpointer, name="face_ai_agent")
    logger.info(
        "agent_graph_compiled",
        extra={
            "nodes": len(graph.get_graph().nodes),
            "checkpointer": type(checkpointer).__name__ if checkpointer else "none",
        },
    )
    return graph


def draw_mermaid(graph: Any | None = None) -> str:
    """Mermaid của graph thật.

    Vẽ từ object đã compile nên sơ đồ luôn đúng với code - không như sơ đồ vẽ tay
    trong tài liệu, thứ luôn lạc hậu sau vài lần refactor.
    """
    graph = graph or build_graph().compile()
    return graph.get_graph().draw_mermaid()
