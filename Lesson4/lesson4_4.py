"""
RAG Tutorial · Agentic RAG với LangGraph
Lesson 4.4 — Agent Loop bằng LangGraph sử dụng init_chat_model (Groq: openai/gpt-oss-20b)
Yêu cầu cài đặt:
pip install -U langchain langchain-groq langgraph langfuse python-dotenv
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError

os.environ.pop("SSL_CERT_FILE", None)
load_dotenv()

# init model and system prompt
SYSTEM_PROMPT = (
    "Bạn là trợ lý kỹ thuật AI chuyên về LangGraph. "
    "BẮT BUỘC sử dụng các công cụ (tools) được cung cấp để tra cứu kiến thức hoặc tính toán chi phí. "
    "Trả lời ngắn gọn, chính xác bằng tiếng Việt dựa trên kết quả từ tool."
)

# Khởi tạo model Groq theo đúng cú pháp yêu cầu
llm = init_chat_model(
    "groq:openai/gpt-oss-20b",
    temperature=0,
)

# init knowledge docs
KNOWLEDGE_DOCS = {
    "trim_messages": (
        "Hàm trim_messages dùng strategy='last', include_system=True, start_on='human' "
        "để giữ lịch sử hội thoại không vượt quá max_tokens."
    ),
    "toolnode": (
        "ToolNode và tools_condition nằm trong langgraph.prebuilt. ToolNode hỗ trợ "
        "thực thi nhiều tool calls song song và tự động bắt lỗi (handle_tool_errors=True)."
    ),
    "recursion_limit": (
        "Trong LangGraph, recursion_limit thay thế cho max_iterations. Khi số bước chạy "
        "vượt quá recursion_limit, đồ thị sẽ ném ngoại lệ GraphRecursionError."
    ),
}

@tool
def search_rag_docs(topic: str) -> str:
    """Tra cứu tài liệu kỹ thuật"""
    key = topic.strip().lower()
    for k, content in KNOWLEDGE_DOCS.items():
        if k in key or key in k:
            return f"[Tài liệu {k}]: {content}"
    return f"Không tìm thấy tài liệu cho chủ đề '{topic}'. Hãy thử từ khóa: trim_messages, toolnode, recursion_limit." 

@tool
def estimate_session_cost(input_tokens: int, output_tokens: int) -> str:
    """Tính chi phí USD của một phiên gọi LLM dựa trên input_tokens và output_tokens (phải là số nguyên dương)."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Lỗi: input_tokens và output_tokens không được là số âm! Hãy đổi sang số dương.")
    cost = (input_tokens / 1000.0) * 0.00015 + (output_tokens / 1000.0) * 0.00060
    return f"Chi phí cho {input_tokens} input tokens và {output_tokens} output tokens là ${cost:.6f} USD."

TOOLS = [search_rag_docs, estimate_session_cost]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

llm_with_tools = llm.bind_tools(TOOLS)


def get_safe_langfuse_callbacks() -> List:
    """
    Kiểm tra biến môi trường của Langfuse. 
    Nếu thiếu key, trả về danh sách rỗng để pipeline vẫn chạy bình thường (Graceful Degradation).
    """
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key or public_key.startswith("pk-lf-placeholder"):
        print("ℹ️  Langfuse keys chưa được cấu hình đầy đủ. Pipeline sẽ chạy KHÔNG có tracing (không làm sập app).")
        return []

    try:
        # Langfuse v3+: `langfuse.callback` đã bị bỏ, handler nằm ở `langfuse.langchain`.
        # Keys/host được cấu hình trên client Langfuse, handler chỉ cần public_key để tìm client đó.
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
        )
        handler = CallbackHandler(public_key=public_key)
        print("✅ Đã kết nối thành công với Langfuse Observability.")
        return [handler]
    except Exception as e:
        print(f"⚠️ Lỗi khởi tạo Langfuse handler: {e}. Bỏ qua tracing.")
        return []


def run_pure_python_agent(user_input: str, max_iterations: int = 5) -> List[BaseMessage]:
    """
    Agent Loop use python
    """
    
    messages: List[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_input)
    ]
    
    for _ in range(max_iterations):
        
        # call llm
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # not tools call
        if not response.tool_calls:
            break
        
        for tc in response.tool_calls:
            tool_fn = TOOLS_BY_NAME[tc["name"]]
            try:
                tool_output = tool_fn.invoke(tc["args"])
            except Exception as err:
                tool_output = f"[Error]: {err}"
                
            messages.append(
                ToolMessage(
                    content=str(tool_output),
                    tool_call_id=tc["id"],
                    name=tc["name"],
                )
            )
            
    return messages

def call_model_node(state: MessagesState, config: RunnableConfig) -> Dict[str, Any]:
    
    """
    Node 'model':
    - Đảm bảo có SystemMessage ở đầu.
    - Rút gọn lịch sử bằng trim_messages (Lesson 3.4) để không vượt ngân sách token.
    - Gọi model Groq đã bind_tools.
    """
    
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
    
    trimmed = trim_messages(
        messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=1500,
        include_system=True,
        start_on="human",
    )
    
    response = llm_with_tools.invoke(trimmed, config=config)
    return {"messages": [response]}

def build_agent_graph():
    
    """
    Xây dựng đồ thị Agent Loop gồm đúng 2 Node ('model' và 'tools')
    và khai báo tập trung toàn bộ các cạnh (Edges) tại đây.
    """
    builder = StateGraph(MessagesState)
    
    # register node
    builder.add_node("model", call_model_node)
    builder.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
    
    # register edge
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", tools_condition)
    builder.add_edge("tools", "model")
    
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


def get_run_config(thread_id: str, recursion_limit: int = 10):
    """
    Tạo config chuẩn chứa:
    - run_name rõ ràng cho Trace.
    - Metadata an toàn (KHÔNG chứa API key, mật khẩu hay PII nhạy cảm).
    - Callback tự động xử lý an toàn khi thiếu key Langfuse.
    """
    safe_callbacks = get_safe_langfuse_callbacks()

    return {
        "configurable": {
            "thread_id": "groq_session_01"  # Bắt buộc phải nằm trong "configurable"
        },
        "recursion_limit": 10
    }
    

if __name__ == "__main__":
    
    if not os.environ.get("GROQ_API_KEY"):
        raise EnvironmentError(
            "Thiếu biến môi trường GROQ_API_KEY! "
            "Hãy chạy: $env:GROQ_API_KEY='gsk_...' hoặc thêm vào file .env"
        )

    print("=" * 80)
    print("CHẠY LESSON 4.4 VỚI MODEL: groq:openai/gpt-oss-20b")
    print("=" * 80)

    agent_graph = build_agent_graph()

    # -----------------------------------------------------------------
    # BÀI TEST 1: SO SÁNH PYTHON THUẦN (4.3) VS LANGGRAPH (4.4)
    # -----------------------------------------------------------------
    query = (
        "Hãy dùng tool tra cứu tài liệu về 'toolnode' và tính chi phí cho "
        "10000 input tokens, 2000 output tokens."
    )

    print(f"\n[1] Đang chạy Bản Python thuần (Lesson 4.3) với câu hỏi:\n    '{query}'")
    py_msgs = run_pure_python_agent(query)
    py_tool_msgs = [m for m in py_msgs if isinstance(m, ToolMessage)]
    py_answer = py_msgs[-1].content

    print(f"\n[2] Đang chạy Bản LangGraph 2 Node (Lesson 4.4) với cùng câu hỏi...")
    config_main = get_run_config(thread_id="groq_session_01", recursion_limit=10)
    lg_state = agent_graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config_main,
    )
    lg_msgs = lg_state["messages"]
    lg_tool_msgs = [m for m in lg_msgs if isinstance(m, ToolMessage)]
    lg_answer = lg_msgs[-1].content

    print("-" * 80)
    print(f"[*] Số lượng Tool được gọi (Python thuần) : {len(py_tool_msgs)}")
    print(f"[*] Số lượng Tool được gọi (LangGraph)    : {len(lg_tool_msgs)}")
    print(f"\n[*] Kết quả Python thuần:\n{py_answer}")
    print(f"\n[*] Kết quả LangGraph 2 Node:\n{lg_answer}")
    print("-" * 80)

    # Kiểm tra cả 2 bản đều gọi cùng số lượng tool và trả về kết quả từ tool giống nhau
    assert len(py_tool_msgs) == len(lg_tool_msgs) and len(lg_tool_msgs) >= 1, (
        "Số lượng tool calls giữa 2 phiên bản không khớp!"
    )
    assert [t.content for t in py_tool_msgs] == [t.content for t in lg_tool_msgs], (
        "Kết quả trả về từ Tool giữa 2 phiên bản không khớp!"
    )

    # -----------------------------------------------------------------
    # BÀI TEST 2: KIỂM TRA MEMORY THEO THREAD_ID (NỐI TIẾP LESSON 3.5)
    # -----------------------------------------------------------------
    print("\n[3] Kiểm tra Memory (Hỏi tiếp câu số 2 trên cùng thread_id='groq_session_01'):")
    followup_state = agent_graph.invoke(
        {"messages": [HumanMessage(content="Chi phí USD vừa tính ở câu trước là bao nhiêu?")]},
        config=config_main,
    )
    print(f"    -> AI trả lời: {followup_state['messages'][-1].content}")

    # -----------------------------------------------------------------
    # BÀI TEST 3: KIỂM TRA GRAPHRECURSIONERROR KHI ĐẶT RECURSION_LIMIT THẤP
    # -----------------------------------------------------------------
    print("\n[4] Kiểm tra chốt chặn recursion_limit = 2 (Cố tình chặn trước khi vòng lặp hoàn tất):")
    # Một chu trình gọi tool cần tối thiểu 3 bước: model (1) -> tools (2) -> model (3).
    # Đặt recursion_limit = 2 sẽ bắt buộc ném GraphRecursionError ngay sau bước 'tools'!
    config_strict = get_run_config(thread_id="groq_session_limit", recursion_limit=2)
    try:
        agent_graph.invoke(
            {"messages": [HumanMessage(content="Hãy tra cứu tài liệu về recursion_limit.")]},
            config=config_strict,
        )
    except GraphRecursionError:
        print("    -> [PASS] Đã bắt được ngoại lệ GraphRecursionError đúng như thiết kế!")

    print("\n" + "=" * 80)
    print("HOÀN TẤT TOÀN BỘ BÀI TẬP LESSON 4.4 VỚI GROQ (openai/gpt-oss-20b)!")
    print("=" * 80)