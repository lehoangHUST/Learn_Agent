from typing import Annotated, List, Literal, Dict, Any
from typing_extensions import TypedDict

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    trim_messages
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.documents import Document

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# init state and budget
MAX_HISTORY_TOKENS = 500
MAX_RETRIES = 2

SYSTEM_PROMPT = (
    "Bạn là Kiến trúc sư AI chuyên về LangGraph và hệ thống RAG. "
    "Chỉ trả lời dựa trên TÀI LIỆU được cung cấp."
)


class AdvancedRAGState(TypedDict):
    # History conversational
    messages: Annotated[List[BaseMessage], add_messages]
    
    # state agent workflow
    active_query: str
    documents: List[Document]
    are_docs_relevant: bool
    draft_answer: str
    is_grounded: bool
    rewrite_count: int
    generation_count: int
    execution_trace: List[str]
    
    
# STORE DATA
VECTOR_DB: List[Document] = [
    Document(
        page_content=(
            "Phương thức add_conditional_edges trong StateGraph giúp tách biệt hoàn toàn "
            "logic điều hướng ra khỏi các Node xử lý, giúp cấu trúc đồ thị tập trung và dễ bảo trì."
        ),
        metadata={"source": "langgraph_architecture.md", "keyword": "add_conditional_edges"},
    ),
    Document(
        page_content=(
            "Hàm trim_messages với strategy='last', include_system=True và start_on='human' "
            "giúp giữ lịch sử hội thoại trong ngân sách max_tokens mà không làm mất SystemMessage "
            "hoặc cắt rời cặp câu hỏi - trả lời."
        ),
        metadata={"source": "lesson_3_4.md", "keyword": "trim_messages"},
    )
]

def safe_trim_history(messages: List[BaseMessage], max_tokens: int = MAX_HISTORY_TOKENS) -> List[BaseMessage]:
    
    msgs = (
        list(messages)
        if messages and isinstance(messages[0], SystemMessage)
        else [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
    )
    
    return trim_messages(
        msgs,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=max_tokens,
        include_system=True,
        start_on="human"
    )
    
def init_turn_node(state: AdvancedRAGState) -> Dict[str, Any]:
    """Khởi tạo lại các bộ đếm cho lượt hỏi mới và lấy câu hỏi gốc"""
    human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    latest_q = human_msgs[-1].content.strip()
    return {
        "active_query": latest_q,
        "rewrite_count": 0,
        "generation_count": 0,
        "are_docs_relevant": False,
        "is_grounded": False,
        "execution_trace": ["init_turn"]
    }
    
def direct_chat_node(state: AdvancedRAGState) -> Dict[str, Any]:
    """Trả lời xã giao trực tiếp không qua RAG"""
    reply = AIMessage(
        content="Xin chào! Tôi là Trợ lý Agentic RAG. Bạn cần hỏi gì về LangGraph hôm nay?"
    )
    return {
        "messages": [reply],
        "execution_trace": state["execution_trace"] + ["direct_chat"],
    }
    
def rewrite_query_node(state: AdvancedRAGState) -> Dict[str, Any]:
    
    trimmed_hist = safe_trim_history(state["messages"][:-1], max_tokens=300)
    current_q = state["active_query"]
    retry = state.get("rewrite_count", 0)
    
    hist_text = " ".join([m.content.lower() for m in trimmed_hist])
    
    if "nó" in current_q.lower() or "phương thức đó" in current_q.lower():
        if "add_conditional_edges" in hist_text:
            rewritten = "Phương thức add_conditional_edges trong StateGraph có ưu điểm gì?"
        else:
            rewritten = "Các tham số của hàm trim_messages trong LangChain hoạt động ra sao?"
    elif retry > 0 and "tách biệt điều hướng" in current_q.lower():
        rewritten = f"{current_q} (từ khóa mở rộng: add_conditional_edges StateGraph)"
    else:
        rewritten = current_q
        
    return {
        "active_query": rewritten,
        "rewrite_count": retry + 1,
        "execution_trace": state["execution_trace"] + [f"rewrite_query(lần_{retry+1})"]
    }

def retrieve_node(state: AdvancedRAGState) -> Dict[str, Any]:
    """Truy xuất dữ liệu từ DB"""
    query_lower = state["active_query"].lower()
    matched_docs = []

    for doc in VECTOR_DB:
        kw = doc.metadata["keyword"]
        if kw in query_lower or any(w in doc.page_content.lower() for w in query_lower.split() if len(w) > 6):
            # Giả lập: Nếu hỏi 'tách biệt điều hướng' mà chưa có từ khóa 'add_conditional_edges' thì chưa khớp
            if "tách biệt điều hướng" in query_lower and "add_conditional_edges" not in query_lower:
                continue
            matched_docs.append(doc)

    return {
        "documents": matched_docs,
        "execution_trace": state["execution_trace"] + ["retrieve"],
    }
    
def grade_documents_node(state: AdvancedRAGState) -> Dict[str, Any]:
    docs = state.get("documents", [])
    
    is_relevant = len(docs) > 0
    return {
        "are_docs_relevant": is_relevant,
        "execution_trace": state["execution_trace"] + [f"grade_docs(relevant={is_relevant})"]
    }
    
def web_fallback_node(state: AdvancedRAGState) -> Dict[str, Any]:
    """Trả lời dự phòng khi kho dữ liệu nội bộ không tìm thấy thông tin sau MAX_RETRIES."""
    fallback_msg = AIMessage(
        content=(
            f"[Web Fallback] Kho nội bộ không có tài liệu về '{state['active_query']}' "
            f"sau {state['rewrite_count']} lần thử tối ưu truy vấn."
        )
    )
    return {
        "messages": [fallback_msg],
        "execution_trace": state["execution_trace"] + ["web_fallback"],
    }


def generate_node(state: AdvancedRAGState) -> Dict[str, Any]:
    """Sinh câu trả lời nháp với lịch sử đã qua trim_messages."""
    trimmed_history = safe_trim_history(state["messages"], max_tokens=MAX_HISTORY_TOKENS)
    docs_text = "\n".join([d.page_content for d in state["documents"]])
    gen_try = state.get("generation_count", 0)

    draft = f"Dựa trên tài liệu: {docs_text} (History rút gọn còn {len(trimmed_history)} msgs)."
    return {
        "draft_answer": draft,
        "generation_count": gen_try + 1,
        "execution_trace": state["execution_trace"] + [f"generate(lần_{gen_try+1})"],
    }


def check_hallucination_node(state: AdvancedRAGState) -> Dict[str, Any]:
    """Kiểm tra ảo giác và nếu hợp lệ thì đẩy câu trả lời vào messages."""
    draft = state["draft_answer"]
    docs_content = " ".join([d.page_content for d in state["documents"]])
    gen_try = state["generation_count"]

    grounded = any(word in draft for word in docs_content.split()[:5])
    updates: Dict[str, Any] = {
        "is_grounded": grounded,
        "execution_trace": state["execution_trace"] + [f"check_hallucination(grounded={grounded})"],
    }

    # Khi đã chuẩn xác (hoặc chạm trần số lần thử), chốt câu trả lời vào lịch sử hội thoại
    if grounded or gen_try >= MAX_RETRIES:
        updates["messages"] = [AIMessage(content=draft)]

    return updates


def route_after_init(state: AdvancedRAGState) -> Literal["chit_chat", "first_turn_rag", "followup_rag"]:
    
    """Định tuyến đầu vào: Xã giao vs. Lượt 1 (tiết kiệm LLM call) vs. Lượt tiếp nối."""
    q_lower = state["active_query"].lower()
    if q_lower in ["xin chào", "hello", "hi", "cảm ơn"]:
        return "chit_chat"

    human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if len(human_msgs) == 1:
        return "first_turn_rag"
    return "followup_rag"

def route_after_grading(
    state: AdvancedRAGState,
) -> Literal["docs_good", "retry_rewrite", "exhausted_fallback"]:
    """Định tuyến sau khi chấm điểm tài liệu."""
    if state["are_docs_relevant"]:
        return "docs_good"
    if state["rewrite_count"] < MAX_RETRIES:
        return "retry_rewrite"
    return "exhausted_fallback"

def route_after_hallucination_check(
    state: AdvancedRAGState,
) -> Literal["pass_to_end", "regenerate"]:
    """Định tuyến sau khi kiểm tra ảo giác."""
    if state["is_grounded"] or state["generation_count"] >= MAX_RETRIES:
        return "pass_to_end"
    return "regenerate"


def build_advanced_rag_graph():
    builder = StateGraph(AdvancedRAGState)
    
    # init node
    builder.add_node("init_turn", init_turn_node)
    builder.add_node("direct_chat", direct_chat_node)
    builder.add_node("rewrite_query", rewrite_query_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade_documents", grade_documents_node)
    builder.add_node("web_fallback", web_fallback_node)
    builder.add_node("generate", generate_node)
    builder.add_node("check_hallucination", check_hallucination_node)
    
    # init edge
    builder.add_edge(START, "init_turn")
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("retrieve", "grade_documents")
    builder.add_edge("generate", "check_hallucination")
    builder.add_edge("direct_chat", END)
    builder.add_edge("web_fallback", END)
    
    # init conditional edges
    builder.add_conditional_edges(
        source="init_turn",
        path=route_after_init,
        path_map={
            "chit_chat": "direct_chat",
            "first_turn_rag": "retrieve",
            "followup_rag": "rewrite_query"
        }
    )
    
    builder.add_conditional_edges(
        source="grade_documents",
        path=route_after_grading,
        path_map={
            "docs_good": "generate",                
            "retry_rewrite": "rewrite_query",
            "exhausted_fallback": "web_fallback",
        },
    )

    builder.add_conditional_edges(
        source="check_hallucination",
        path=route_after_hallucination_check,
        path_map={
            "pass_to_end": END,                 
            "regenerate": "generate",
        },
    )
    
    # compile with checkpointer
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


if __name__ == "__main__":
    app = build_advanced_rag_graph()
    config = {"configurable": {"thread_id": "clean_architecture_session"}}

    test_cases = [
        (
            "Lượt 1 (Câu hỏi rõ nghĩa -> Đi thẳng tới retrieve)",
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content="Hàm trim_messages dùng làm gì?")],
        ),
        (
            "Lượt 2 (Câu hỏi mơ hồ -> Vòng lặp grade_docs quay lại rewrite_query lần 2)",
            [HumanMessage(content="Làm sao để tách biệt điều hướng trong đồ thị?")],
        ),
        (
            "Lượt 3 (Câu hỏi tiếp nối dùng đại từ 'phương thức đó')",
            [HumanMessage(content="Phương thức đó có ưu điểm gì?")],
        ),
        (
            "Lượt 4 (Ngoài phạm vi -> Thử tối đa 2 lần rồi chuyển qua web_fallback)",
            [HumanMessage(content="Thời tiết Hà Nội hôm nay thế nào?")],
        ),
    ]

    print("=" * 85)
    print("KIỂM THỬ LANGGRAPH VỚI TOÀN BỘ EDGE TẬP TRUNG TRONG HÀM BUILD_GRAPH()")
    print("=" * 85)

    for label, msgs in test_cases:
        out = app.invoke({"messages": msgs}, config=config)
        path_str = " -> ".join(out["execution_trace"])
        print(f"\n[+] {label}")
        print(f"    Luồng chạy : START -> {path_str} -> END")
        print(f"    Query cuối : '{out['active_query']}'")
        print(f"    AI trả lời : {out['messages'][-1].content}")