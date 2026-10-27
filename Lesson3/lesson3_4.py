"""
RAG Tutorial · Retrieval · Agentic RAG với LangGraph
"""

import os
from dotenv import load_dotenv
from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# init config settings
os.environ.pop("SSL_CERT_FILE", None)
load_dotenv()

# Settings budget token and price model (cost config)
HISTORY_MAX_TOKENS = 600 # Budget max for system + history conversational
REWRITE_MAX_TOKENS = 500 # Budget for write query

# Model chính phải có context window lớn: nhánh baseline (enable_trimming=False)
# cố tình để lịch sử phình to. allam-2-7b chỉ có 4096 tokens nên Groq trả về
# lỗi context_length_exceeded ngay ở node rewrite_or_pass.
MODEL_NAME = "groq:openai/gpt-oss-20b"   # 131k context

# Chặn độ dài câu trả lời để lịch sử không phình quá nhanh qua từng lượt
MAX_OUTPUT_TOKENS = 512

# Trần an toàn cho nhánh KHÔNG rút gọn: vẫn để lịch sử phình to để so sánh,
# nhưng cắt trước khi vượt context window của model (tránh crash giữa demo).
UNBOUNDED_SAFETY_MAX_TOKENS = 20000

# Bảng giá giả định (GPT-4o-mini: $0.15 / 1M input tokens, $0.60 / 1M output tokens)
PRICE_PER_1K_INPUT = 0.00015
PRICE_PER_1K_OUTPUT = 0.00060


SYSTEM_PROMPT = (
    "Bạn là trợ lý kỹ thuật chuyên về LangGraph và Agentic RAG. "
    "Hãy trả lời chính xác dựa trên TÀI LIỆU KỸ THUẬT được cung cấp. "
    "Nếu tài liệu không có thông tin, hãy nói rõ là không biết."
)

# CSDL (MOCK VECTOR STORE for phase 2 & 3)
KNOWLEDGE_BASE: List[Document] = [
    Document(
        page_content=(
            "Hàm trim_messages trong langchain_core.messages dùng để cắt gọn lịch sử hội thoại "
            "theo ngân sách max_tokens. Các tham số quan trọng gồm: strategy='last' (giữ tin mới nhất), "
            "include_system=True (bảo toàn SystemMessage), và start_on='human' (tránh cắt lệch cặp "
            "Human/AI hoặc làm mồ côi ToolMessage)."
        ),
        metadata={"source": "lesson_3_4.md", "topic": "trim_messages"},
    ),
    Document(
        page_content=(
            "Để đếm token trong LangChain: Trước khi gọi LLM, dùng count_tokens_approximately "
            "hoặc tokenizer của model để ước lượng nhanh. Sau khi gọi LLM, đọc thuộc tính "
            "usage_metadata trên AIMessage trả về để lấy số input_tokens và output_tokens thực tế."
        ),
        metadata={"source": "lesson_3_4.md", "topic": "token_counting"},
    ),
    Document(
        page_content=(
            "Trong LangGraph, có 3 loại dữ liệu: Conversation history (lịch sử tin nhắn trong 1 session), "
            "Application state (dữ liệu điều phối của workflow), và Long-term memory (bộ nhớ xuyên "
            "session sẽ học ở Lesson 5.9). Ngoài ra SummarizationMiddleware của create_agent (Lesson 4.5) "
            "hỗ trợ tóm tắt tự động lịch sử cũ thành summary message."
        ),
        metadata={"source": "lesson_3_4.md", "topic": "data_types_and_summary"},
    )
]

def retrieve_documents(query: str) -> List[Document]:
    
    query_lower = query.lower()
    scored_docs = []
    for doc in KNOWLEDGE_BASE:
        content_lower = doc.page_content.lower()
        score = 0
        keywords = [
            "trim_messages", "start_on", "include_system", "strategy",
            "usage_metadata", "count_tokens_approximately", "đếm token", "token",
            "ba loại dữ liệu", "long-term memory", "summarizationmiddleware", "tóm tắt"
        ]
        for kw in keywords:
            if kw in query_lower and kw in content_lower:
                score += 3
                
        for word in query_lower.split():
            if len(word) > 3 and word in content_lower:
                score += 1
        if score > 0:
            scored_docs.append((score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:2]] or [KNOWLEDGE_BASE[0]]


class MockChatModel:
    
    def invoke(self, messages: List[BaseMessage], config: Optional[RunnableConfig] = None) -> AIMessage:
        input_tokens = count_tokens_approximately(messages)
        last_msg_content = messages[-1].content.lower()

        # Phân biệt lời gọi Rewrite Query hay Generate Answer
        if "viết lại câu hỏi" in last_msg_content or "standalone question" in last_msg_content:
            # Xử lý contextualize câu hỏi tiếp nối
            full_text = " ".join([m.content.lower() for m in messages])
            if "start_on" in last_msg_content or "tham số" in last_msg_content or "nó" in last_msg_content:
                if "trim_messages" in full_text:
                    reply = "Các tham số start_on và include_system trong hàm trim_messages có tác dụng gì?"
                else:
                    reply = "Hàm đếm token trước và sau khi gọi LLM bằng usage_metadata và count_tokens_approximately hoạt động thế nào?"
            elif "lesson 4.5" in last_msg_content or "tóm tắt" in last_msg_content:
                reply = "SummarizationMiddleware ở Lesson 4.5 và 3 loại dữ liệu Long-term memory trong LangGraph là gì?"
            else:
                reply = messages[-1].content
        else:
            # Xử lý sinh câu trả lời RAG
            if "trim_messages" in last_msg_content:
                reply = (
                    "Hàm `trim_messages` sử dụng `strategy='last'` để giữ tin nhắn mới nhất trong ngân sách "
                    "`max_tokens`. Tham số `include_system=True` giữ lại SystemMessage, còn `start_on='human'` "
                    "đảm bảo tin nhắn đầu tiên sau System luôn là HumanMessage để không cắt lệch cặp hội thoại."
                )
            elif "usage_metadata" in last_msg_content or "đếm token" in last_msg_content:
                reply = (
                    "Trước khi gọi LLM, bạn dùng `count_tokens_approximately` hoặc tokenizer để ước lượng. "
                    "Sau khi gọi, bạn lấy số token thực tế từ thuộc tính `usage_metadata` của `AIMessage`."
                )
            else:
                reply = (
                    "LangGraph chia dữ liệu thành 3 loại: Conversation history (trong 1 session), "
                    "Application state (dữ liệu workflow), và Long-term memory (xuyên session - Lesson 5.9). "
                    "Để tóm tắt tự động có thể dùng `SummarizationMiddleware` (Lesson 4.5)."
                )

        output_tokens = count_tokens_approximately([AIMessage(content=reply)])
        return AIMessage(
            content=reply,
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        )
        
def get_llm_chat():
    
    llm_chat = init_chat_model(
        MODEL_NAME,
        temperature=0,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    return llm_chat

llm_chat = get_llm_chat()

def trim_conversation_history(
    messages: List[BaseMessage],
    max_tokens: int = HISTORY_MAX_TOKENS,
) -> List[BaseMessage]:
    """
    Áp dụng chiến lược rút gọn lịch sử hội thoại theo ngân sách token.
    Đảm bảo 3 điều kiện Pass của Lesson 3.4:
      1. Không vượt ngân sách max_tokens.
      2. Không mất SystemMessage (include_system=True).
      3. Không cắt lệch cặp Human/AI hoặc Tool call (start_on="human").
    """
    # Đảm bảo luôn có SystemMessage ở đầu trước khi trim
    if not messages or not isinstance(messages[0], SystemMessage):
        messages_with_sys = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
    else:
        messages_with_sys = list(messages)

    trimmed = trim_messages(
        messages_with_sys,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=max_tokens,
        include_system=True,
        start_on="human",
        allow_partial=False,
    )
    return trimmed

def validate_trimmed_messages(trimmed: List[BaseMessage], max_tokens: int) -> None:
    estimated_tokens = count_tokens_approximately(trimmed)
    assert estimated_tokens <= max_tokens, (
        f"[FAIL] Vượt ngân sách token: {estimated_tokens} > {max_tokens}"
    )

    assert len(trimmed) >= 1 and isinstance(trimmed[0], SystemMessage), (
        "[FAIL] Mất SystemMessage ở đầu danh sách!"
    )

    non_system = trimmed[1:]
    if non_system:
        assert isinstance(non_system[0], HumanMessage), (
            f"[FAIL] Cắt lệch cặp message! Message đầu tiên sau System là {type(non_system[0]).__name__}"
        )
        for idx, msg in enumerate(non_system):
            if isinstance(msg, ToolMessage):
                assert idx > 0 and isinstance(
                    non_system[idx - 1], (AIMessage, ToolMessage)
                ), "[FAIL] ToolMessage bị cắt rời khỏi AIMessage gọi tool!"
                

class TurnMetrics(TypedDict):
    turn_index: int
    llm_calls_in_turn: int
    raw_history_tokens: int
    trimmed_history_tokens: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    rewritten_query: str


class AgenticRAGState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    enable_trimming: bool
    rewritten_query: str
    documents: List[Document]
    turn_metrics: List[TurnMetrics]
    current_turn_llm_calls: int
    current_turn_input_tokens: int
    current_turn_output_tokens: int
    
    
def rewrite_or_pass_node(state: AgenticRAGState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Node 1: Xử lý câu hỏi đầu vào.
    - Điều kiện Pass Lesson 3.3: Nếu là lượt đầu tiên (chỉ có 1 HumanMessage),
      KHÔNG gọi LLM rewriter (tốn 0 LLM call phụ), đi thẳng tới bước Retrieve.
    - Nếu là câu hỏi tiếp nối (lượt >= 2), dùng lịch sử đã rút gọn để viết lại câu hỏi độc lập.
    """
    messages = state["messages"]
    human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
    latest_question = human_msgs[-1].content

    # LƯỢT ĐẦU TIÊN -> Bỏ qua LLM Rewrite để tiết kiệm tuyệt đối lời gọi LLM
    if len(human_msgs) == 1:
        return {
            "rewritten_query": latest_question,
            "current_turn_llm_calls": 0,
            "current_turn_input_tokens": 0,
            "current_turn_output_tokens": 0,
        }

    if state.get("enable_trimming", True):
        history_for_rewrite = trim_conversation_history(messages[:-1], max_tokens=REWRITE_MAX_TOKENS)
    else:
        # "Unbounded" nhưng vẫn chặn ở trần an toàn của context window
        history_for_rewrite = trim_conversation_history(
            messages[:-1], max_tokens=UNBOUNDED_SAFETY_MAX_TOKENS
        )

    rewrite_prompt = history_for_rewrite + [
        HumanMessage(
            content=(
                "Dựa vào lịch sử hội thoại trên, hãy viết lại câu hỏi tiếp nối sau đây "
                "thành một câu hỏi độc lập (standalone question) đầy đủ ngữ cảnh để tìm kiếm tài liệu.\n"
                "CHỈ trả về duy nhất câu hỏi đã viết lại, giữ nguyên tên hàm/tham số gốc, "
                "không giải thích, không thêm lời mở đầu.\n"
                f"Câu hỏi tiếp nối: {latest_question}"
            )
        )
    ]

    response = llm_chat.invoke(rewrite_prompt, config=config)
    usage = getattr(response, "usage_metadata", {}) or {}

    return {
        "rewritten_query": response.content.strip(),
        "current_turn_llm_calls": 1,
        "current_turn_input_tokens": usage.get("input_tokens", 0),
        "current_turn_output_tokens": usage.get("output_tokens", 0),
    }
    
def retrieve_node(state: AgenticRAGState) -> Dict[str, Any]:
    """Node 2: Truy xuất tài liệu từ câu hỏi đã được chuẩn hóa."""
    query = state["rewritten_query"]
    docs = retrieve_documents(query)
    return {"documents": docs}

def generate_node(state: AgenticRAGState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Node 3: Sinh câu trả lời với lịch sử được kiểm soát bởi trim_messages.
    """
    messages = state["messages"]
    enable_trimming = state.get("enable_trimming", True)

    # Chuẩn bị lịch sử hội thoại có SystemMessage
    full_with_sys = (
        list(messages)
        if isinstance(messages[0], SystemMessage)
        else [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
    )
    raw_history_tokens = count_tokens_approximately(full_with_sys)

    if enable_trimming:
        active_messages = trim_conversation_history(full_with_sys, max_tokens=HISTORY_MAX_TOKENS)
        # Kiểm tra nghiêm ngặt 3 điều kiện Pass của Lesson 3.4 ngay trong runtime
        validate_trimmed_messages(active_messages, max_tokens=HISTORY_MAX_TOKENS)
    else:
        # Giữ nguyên tinh thần "không rút gọn", chỉ chặn ở trần an toàn context window
        active_messages = trim_conversation_history(
            full_with_sys, max_tokens=UNBOUNDED_SAFETY_MAX_TOKENS
        )

    trimmed_history_tokens = count_tokens_approximately(active_messages)

    # Ghép Context RAG vào câu hỏi cuối cùng (không làm hỏng cấu trúc lịch sử)
    docs_context = "\n\n".join(
        [f"[Tài liệu {i+1}]: {d.page_content}" for i, d in enumerate(state["documents"])]
    )
    last_human = active_messages[-1]
    rag_augmented_human = HumanMessage(
        content=f"TÀI LIỆU KỸ THUẬT:\n{docs_context}\n\nCÂU HỎI: {last_human.content}"
    )

    prompt_messages = active_messages[:-1] + [rag_augmented_human]

    # Gọi LLM sinh câu trả lời
    response = llm_chat.invoke(prompt_messages, config=config)
    usage = getattr(response, "usage_metadata", {}) or {}

    in_tok = state.get("current_turn_input_tokens", 0) + usage.get("input_tokens", 0)
    out_tok = state.get("current_turn_output_tokens", 0) + usage.get("output_tokens", 0)
    total_tok = in_tok + out_tok
    llm_calls = state.get("current_turn_llm_calls", 0) + 1

    cost_usd = (in_tok / 1000.0) * PRICE_PER_1K_INPUT + (out_tok / 1000.0) * PRICE_PER_1K_OUTPUT

    existing_metrics = list(state.get("turn_metrics") or [])
    new_metric: TurnMetrics = {
        "turn_index": len(existing_metrics) + 1,
        "llm_calls_in_turn": llm_calls,
        "raw_history_tokens": raw_history_tokens,
        "trimmed_history_tokens": trimmed_history_tokens,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": total_tok,
        "cost_usd": cost_usd,
        "rewritten_query": state["rewritten_query"],
    }
    existing_metrics.append(new_metric)

    return {
        "messages": [response],
        "turn_metrics": existing_metrics,
    }
    
def builder_rag_graph():
    
    builder = StateGraph(AgenticRAGState)
    # add node
    builder.add_node("rewrite_or_pass", rewrite_or_pass_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    
    # add edge
    builder.add_edge(START, "rewrite_or_pass")
    builder.add_edge("rewrite_or_pass", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
      
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

def get_langfuse_config(session_id: str, thread_id: str) -> RunnableConfig:
    """
    Tạo RunnableConfig gắn session_id cho Langfuse để theo dõi trên Sessions View.
    Nếu không cấu hình LANGFUSE_PUBLIC_KEY, trả về config tiêu chuẩn của LangGraph.
    """
    config: RunnableConfig = {
        "configurable": {"session_id": session_id, "thread_id": thread_id}
    }

    if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
        try:
            from langfuse.langchain import CallbackHandler
            langfuse_handler = CallbackHandler()
            
            config = {
                "configurable": {"session_id": session_id, "thread_id": thread_id},
                "callbacks": [langfuse_handler],
                "metadata": {
                    "langfuse_session_id": session_id,
                    "langfuse_tags": ["lesson-3.3", "contextual-rag"],
                },
            }
        except ImportError:
            pass
    return config


def run_simulation(enable_trimming: bool, session_name: str) -> Dict[str, Any]:
    """Chạy một session hội thoại nhiều lượt (Multi-turn) để đo token & chi phí."""
    graph = builder_rag_graph()
    config = get_langfuse_config(session_id=session_name, thread_id=session_name)

    # Khởi tạo SystemMessage trong state ban đầu
    initial_messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

    # Kịch bản 5 lượt hội thoại (bao gồm câu hỏi gốc + câu hỏi tiếp nối dùng đại từ)
    user_turns = [
        "Hàm trim_messages trong LangChain dùng để làm gì?",
        "Các tham số start_on và include_system của nó hoạt động ra sao?",
        "Làm sao để đếm token trước và sau khi gọi LLM?",
        "Vậy còn 3 loại dữ liệu trong LangGraph gồm những gì?",
        "Nếu tôi muốn tóm tắt lịch sử tự động ở Lesson 4.5 thì dùng công cụ nào?",
    ]

    final_state = None
    for idx, question in enumerate(user_turns):
        input_msgs = (
            initial_messages + [HumanMessage(content=question)]
            if idx == 0
            else [HumanMessage(content=question)]
        )
        final_state = graph.invoke(
            {
                "messages": input_msgs,
                "enable_trimming": enable_trimming,
            },
            config=config,
        )

    return final_state


def test_edge_case_tool_call_integrity():
    """
    Kiểm thử bổ sung: Đảm bảo không cắt lệch cặp AIMessage (có tool_calls) và ToolMessage.
    """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="Câu hỏi rất dài ở lượt 1 " * 15),
        AIMessage(
            content="",
            tool_calls=[{"name": "search_docs", "args": {"q": "trim"}, "id": "call_1"}],
        ),
        ToolMessage(content="Kết quả tool trả về " * 10, tool_call_id="call_1"),
        AIMessage(content="Câu trả lời hoàn chỉnh lượt 1 " * 10),
        HumanMessage(content="Câu hỏi ngắn ở lượt 2?"),
    ]
    trimmed = trim_conversation_history(messages, max_tokens=120)
    validate_trimmed_messages(trimmed, max_tokens=120)
    return trimmed


if __name__ == "__main__":
    print("=" * 78)
    print("BẮT ĐẦU KIỂM THỬ LESSON 3.4: CONTEXT WINDOW & CHIẾN LƯỢC RÚT GỌN")
    print("=" * 78)

    # 1. Chạy thực nghiệm TRƯỚC khi rút gọn (Unbounded History)
    state_before = run_simulation(enable_trimming=False, session_name="session-before-trim")
    metrics_before: List[TurnMetrics] = state_before["turn_metrics"]

    # 2. Chạy thực nghiệm SAU khi rút gọn (Trimmed History với max_tokens=HISTORY_MAX_TOKENS)
    state_after = run_simulation(enable_trimming=True, session_name="session-after-trim")
    metrics_after: List[TurnMetrics] = state_after["turn_metrics"]

    # 3. Kiểm thử edge-case với ToolMessage
    test_edge_case_tool_call_integrity()

    # In bảng so sánh từng lượt (tương ứng với Langfuse Sessions View)
    print(f"\nNgân sách History Token đặt ra (HISTORY_MAX_TOKENS): {HISTORY_MAX_TOKENS} tokens")
    print("-" * 78)
    print(
        f"{'Lượt':<5} | {'LLM Calls':<9} | {'Hist(Trước)':<12} | {'Hist(Sau)':<10} | "
        f"{'Total Tok(Trước -> Sau)':<24} | {'Tiết kiệm'}"
    )
    print("-" * 78)

    total_cost_before = 0.0
    total_cost_after = 0.0

    for m_bef, m_aft in zip(metrics_before, metrics_after):
        turn = m_aft["turn_index"]
        calls = m_aft["llm_calls_in_turn"]
        h_bef = m_bef["raw_history_tokens"]
        h_aft = m_aft["trimmed_history_tokens"]
        t_bef = m_bef["total_tokens"]
        t_aft = m_aft["total_tokens"]
        saved_pct = ((t_bef - t_aft) / t_bef * 100) if t_bef > 0 else 0.0

        total_cost_before += m_bef["cost_usd"]
        total_cost_after += m_aft["cost_usd"]

        print(
            f"#{turn:<4} | {calls:<9} | {h_bef:<12} | {h_aft:<10} | "
            f"{t_bef:<9} -> {t_aft:<11} | {saved_pct:>6.1f}%"
        )

    print("-" * 78)
    cost_saved_pct = (
        (total_cost_before - total_cost_after) / total_cost_before * 100
        if total_cost_before > 0
        else 0.0
    )
    print(f"Tổng Cost Session (Trước rút gọn): ${total_cost_before:.6f}")
    print(f"Tổng Cost Session (Sau rút gọn)  : ${total_cost_after:.6f} (Giảm {cost_saved_pct:.1f}%)")
    print("-" * 78)

    # =================================================================
    # XÁC MINH TẤT CẢ CÁC ĐIỀU KIỆN PASS (ASSERTIONS)
    # =================================================================
    # ĐK 1 (Lesson 3.3): Lượt đầu không tốn thêm lời gọi LLM nào (chỉ đúng 1 call cho generate)
    assert metrics_after[0]["llm_calls_in_turn"] == 1, "Lượt đầu tiên chỉ được gọi LLM đúng 1 lần!"

    # ĐK 2 (Lesson 3.3): Câu hỏi tiếp nối ("nó") ở lượt 2 được viết lại chứa từ khóa "trim_messages"
    rewritten_turn_2 = metrics_after[1]["rewritten_query"].lower()
    assert "trim_messages" in rewritten_turn_2, (
        f"Câu hỏi tiếp nối chưa được contextualize đúng: {rewritten_turn_2}"
    )

    # ĐK 3 (Lesson 3.4): Không bao giờ vượt ngân sách token đã đặt ở mọi lượt
    for m in metrics_after:
        assert m["trimmed_history_tokens"] <= HISTORY_MAX_TOKENS, (
            f"Lượt {m['turn_index']} vượt ngân sách token!"
        )

    print("\n[PASS ALL CHECKS]:")
    print("  1. Lượt đầu không tốn thêm lời gọi LLM nào (1 LLM call duy nhất).")
    print(f"  2. Câu hỏi tiếp nối lượt 2 được viết lại chuẩn: '{metrics_after[1]['rewritten_query']}'")
    print(f"  3. Mọi lượt hội thoại đều giữ History <= {HISTORY_MAX_TOKENS} tokens.")
    print("  4. SystemMessage luôn được giữ lại (include_system=True).")
    print("  5. Không cắt lệch cặp Human/AI hoặc ToolMessage (start_on='human').")