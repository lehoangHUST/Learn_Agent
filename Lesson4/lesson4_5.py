"""
Lesson 4.5 — create_agent & middleware (LangChain 1.x)
======================================================
File code hoàn chỉnh minh họa:
1. create_agent API với response_format (ToolStrategy / ProviderStrategy),
   Memory (checkpointer, store) và context_schema.
2. Hệ thống Middleware:
   - Built-in middleware: ModelCallLimitMiddleware, ToolCallLimitMiddleware, PIIMiddleware
   - Custom middleware dạng Decorator (@before_model, @wrap_model_call, @wrap_tool_call)
   - Custom middleware dạng Class (AgentMiddleware subclass)
3. So sánh 3 cách cài đặt Agent (Raw Tool-calling Loop vs StateGraph vs create_agent).
4. Tích hợp Langfuse CallbackHandler để trace toàn bộ vòng đời agent và các bước middleware.

Chế độ chạy:
- Mặc định script sử dụng `FakeListChatModel` / Mock ChatModel thông minh hỗ trợ tool calls
  để bạn có thể chạy test luồng (dry-run) ngay lập tức mà không tốn API token.
- Nếu có biến môi trường `OPENAI_API_KEY`, bạn có thể đổi `USE_REAL_LLM = True` để gọi model thật.
"""

from __future__ import annotations

import os
import time
import logging
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Any, Callable, Annotated, Sequence
from typing_extensions import TypedDict

from pydantic import BaseModel, Field

# LangChain Core & Messages
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain.chat_models import init_chat_model

# Langgraph
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime
from langgraph.types import Command

# Langchain agent * Middleware
try:
    from langchain.agents import create_agent, AgentState
    from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
    from langchain.agents.middleware import (
        AgentMiddleware,
        ModelRequest,
        ModelResponse,
        before_agent,
        before_model,
        after_model,
        after_agent,
        wrap_model_call,
        wrap_tool_call,
        dynamic_prompt,
        ModelCallLimitMiddleware,
        ToolCallLimitMiddleware,
        PIIMiddleware,
    )
    LANGCHAIN_V1_AVAILABLE = True
except ImportError:
    LANGCHAIN_V1_AVAILABLE = False

print("Check LANGCHAIN_V1_AVAILABLE: ", LANGCHAIN_V1_AVAILABLE)

# Langfuse Tracing
try:
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
    LANGFUSE_AVAILABLE = True
except ImportError:
    try:
        from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
        LANGFUSE_AVAILABLE = True
    except ImportError:
        LANGFUSE_AVAILABLE = False
        
os.environ.pop("SSL_CERT_FILE", None)
load_dotenv()    

        
# Cấu hình Logging rõ ràng để quan sát thứ tự chạy của Middleware
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Lesson4.5")


def get_chat_model():
    llm_chat = init_chat_model(
        "groq:openai/gpt-oss-20b",
        temperature=0,
    )
    return llm_chat

########
# 1. Init tools and context schema, structured output
########
@tool
def get_weather(city: str) -> str:
    """Tra cứu thời tiết hiện tại của một thành phố"""
    time.sleep(0.05) # giả lập độ trể I/O ghi gọi từ API lấy nhiệt độ từ bên thứ 3
    return f"Thời tiết tại {city}: 26.5°C, trời mát, độ ẩm 70%."

@tool
def calculate_tax(amount: float, rate: float = 0.1) -> str:
    """Tính thuế dựa trên số tiền và thuế suất"""
    time.sleep(0.03)
    tax = amount * rate
    return f"Số tiền: {amount}, Thuế ({rate*100:.0f}%): {tax:.2f}, Tổng: {amount + tax:.2f}"


TOOLS = [get_weather, calculate_tax]

@dataclass
class UserContext:
    """Schema định nghĩa ngữ cảnh (context_schema) truyền vào agent lúc runtime."""
    user_id: str
    user_tier: str = "free"  # "free" hoặc "pro"
    language: str = "vi"


class WeatherReport(BaseModel):
    """Schema đầu ra có cấu trúc (response_format) cho câu trả lời cuối cùng."""
    city: str = Field(description="Tên thành phố")
    temperature_c: float = Field(description="Nhiệt độ theo độ C")
    summary: str = Field(description="Tóm tắt tình hình thời tiết")

if LANGCHAIN_V1_AVAILABLE:

    # --- 2.1. Middleware tự viết dạng Decorator ---
    @before_model
    def guardrail_and_trim_middleware(
        state: AgentState, runtime: Runtime[UserContext]
    ) -> dict[str, Any] | Command | None:
        """
        Node-style hook: Chạy trước mỗi lần gọi LLM.
        - Kiểm tra guardrail (chặn từ khóa nhạy cảm).
        - Kiểm tra số lượng message trong state.
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        if isinstance(last_msg, HumanMessage) and "DROP TABLE" in str(last_msg.content).upper():
            logger.warning("[Guardrail @before_model] Phát hiện SQL Injection! Dừng agent ngay lập tức.")
            return Command(
                update={
                    "messages": [
                        AIMessage(content="Yêu cầu bị từ chối vì vi phạm chính sách an toàn.")
                    ]
                },
                goto="__end__",
            )

        user_id = runtime.context.user_id if runtime and runtime.context else "anonymous"
        logger.info(
            f"[Hook @before_model] Chuẩn bị gọi model | user_id={user_id} | tổng số messages={len(messages)}"
        )
        return None

    @wrap_model_call
    def dynamic_prompt_and_timing_middleware(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """
        Wrap-style hook quanh lệnh gọi Model:
        - Tiêm thông tin từ `runtime.context` vào system prompt.
        - Đo thời gian phản hồi (latency) của từng lần gọi model.
        """
        t0 = time.perf_counter()
        ctx: UserContext | None = getattr(request.runtime, "context", None)

        if ctx is not None:
            extra_instruction = (
                f"\n[Runtime Context] Đang hỗ trợ user_id={ctx.user_id} "
                f"(Hạng: {ctx.user_tier}, Ngôn ngữ: {ctx.language})."
            )
            current_sys = request.system_message.content if request.system_message else ""
            request = request.override(
                system_message=SystemMessage(content=f"{current_sys}{extra_instruction}")
            )

        logger.info("[Hook @wrap_model_call] >>> Bắt đầu gọi LLM...")
        response = handler(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"[Hook @wrap_model_call] <<< LLM phản hồi sau {elapsed_ms:.2f} ms")
        return response

    @wrap_tool_call
    def tool_audit_log_middleware(
        request: Any,
        handler: Callable[[Any], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """
        Wrap-style hook quanh lệnh gọi Tool:
        - Ghi log tên tool, tham số đầu vào, thời gian chạy và kết quả trả về.
        """
        tool_call = request.tool_call
        tool_name = tool_call.get("name", "unknown_tool")
        tool_args = tool_call.get("args", {})

        logger.info(f"[Hook @wrap_tool_call] >>> Thực thi tool '{tool_name}' với args={tool_args}")
        t0 = time.perf_counter()
        try:
            result = handler(request)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                f"[Hook @wrap_tool_call] <<< Tool '{tool_name}' hoàn tất trong {elapsed_ms:.2f} ms"
            )
            return result
        except Exception as exc:
            logger.error(f"[Hook @wrap_tool_call] !!! Lỗi khi chạy tool '{tool_name}': {exc}")
            raise

    # --- 2.2. Middleware tự viết dạng Class (Full Lifecycle Audit) ---
    class ExecutionAuditMiddleware(AgentMiddleware):
        """
        Class-based Middleware quản lý trạng thái thống kê xuyên suốt 1 lần chạy (run)
        bằng cách kết hợp cả `before_agent`, `after_model`, và `after_agent`.
        """

        def __init__(self, agent_tag: str = "v1-production") -> None:
            super().__init__()
            self.agent_tag = agent_tag
            self._run_start_time: float = 0.0
            self.model_Empty_or_Tool_steps: int = 0

        def before_agent(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:
            self._run_start_time = time.perf_counter()
            self.model_Empty_or_Tool_steps = 0
            logger.info(f"[ClassMiddleware:before_agent] Khởi động phiên chạy ({self.agent_tag})")
            return None

        def after_model(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:
            self.model_Empty_or_Tool_steps += 1
            last_msg = state["messages"][-1]
            tool_calls = getattr(last_msg, "tool_calls", [])
            logger.info(
                f"[ClassMiddleware:after_model] Bước suy luận #{self.model_Empty_or_Tool_steps} "
                f"| Số tool_calls sinh ra: {len(tool_calls)}"
            )
            return None

        def after_agent(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:
            total_ms = (time.perf_counter() - self._run_start_time) * 1000
            logger.info(
                f"[ClassMiddleware:after_agent] Kết thúc phiên ({self.agent_tag}) | "
                f"Tổng số bước gọi model: {self.model_Empty_or_Tool_steps} | "
                f"Tổng thời gian: {total_ms:.2f} ms"
            )
            return None


def run_way1_manual_loop(query: str) -> list[BaseMessage]:
    # Tự quản lý danh sách messages tự bind tools và viết vòng lặp while
    model = get_chat_model()
    model_with_tools = model.bind_tools(TOOLS)
    tool_map = {t.name: t for t in TOOLS}
    
    messages: list[BaseMessage] = [
        SystemMessage(content="Bạn là trợ lý hữu ích."),
        HumanMessage(content=query)
    ]
    
    max_iterations = 4
    for step in range(1, max_iterations + 1):
        logger.info(f"[Way 1 - Manual] Bước {step}: Gọi model...")
        ai_msg: AIMessage = model_with_tools.invoke(messages)
        messages.append(ai_msg)
        
        if not ai_msg.tool_calls:
            logger.info("[Way 1 - Manual] Model không gọi thêm tool. Kết thúc.")
            break
        
        for tc in ai_msg.tool_calls:
            t_name = tc["name"]
            t_args = tc["args"]
            logger.info(f"[Way 1 - Manual] Thực thi tool {t_name}({t_args})")
            tool_obj = tool_map[t_name]
            tool_output = tool_obj.invoke(t_args)
            messages.append(
                ToolMessage(content=str(tool_output), tool_call_id=tc["id"], name=t_name)
            )
    print(f"-> Kết quả Cách 1: {messages[-1].content}")
    return messages

def run_way2_langgraph(query: str) -> dict[str, Any]:
    
    model = get_chat_model().bind_tools(TOOLS)
    
    def call_model_node(state: MessagesState) -> dict[str, list[BaseMessage]]:
            logger.info(f"[Way 2 - StateGraph] Node 'model' chạy với {len(state['messages'])} messages")
            response = model.invoke(state["messages"])
            return {"messages": [response]}
    
    builder = StateGraph(MessagesState)
    builder.add_node("model", call_model_node)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "model")

    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "thread-way-2"}}
    result = graph.invoke({"messages": [HumanMessage(content=query)]}, config=config)
    print(f"-> Kết quả Cách 2: {result['messages'][-1].content}")
    return result

def run_way3_create_agent_with_middleware(query: str) -> dict[str, Any]:
    
    if not LANGCHAIN_V1_AVAILABLE:
        raise RuntimeError(
            "Môi trường hiện tại chưa cài đặt `langchain>=1.0.0`. "
            "Hãy chạy: `pip install -U langchain langgraph langfuse`"
        )
        
    model = get_chat_model()
    checkpointer = InMemorySaver()
    store = InMemoryStore()
    
    model_limiter = ModelCallLimitMiddleware(
        run_limit=3,
        thread_limit=10,
        exit_behavior="end",
    )
    tool_limiter = ToolCallLimitMiddleware(
        run_limit=2,
        exit_behavior="continue",
    )
    pii_protector = PIIMiddleware(
        "email",
        strategy="redact",
        apply_to_input=True,
    )
    
    agent = create_agent(
        model=model,
        tools=TOOLS,
        system_prompt="Bạn là trợ lý AI chuyên nghiệp hỗ trợ tra cứu thời tiết và tính thuế.",
        middleware=[
            ExecutionAuditMiddleware(agent_tag="lesson-4.5-demo"),
            pii_protector,
            model_limiter,
            tool_limiter,
            guardrail_and_trim_middleware,
            dynamic_prompt_and_timing_middleware,
            tool_audit_log_middleware,
        ],
        context_schema=UserContext,
        checkpointer=checkpointer,
        store=store,
        response_format=ToolStrategy(WeatherReport),
        name="weather_tax_assistant",
    )
    
    callbacks: list[Any] = []
    if LANGFUSE_AVAILABLE and os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        langfuse_handler = LangfuseCallbackHandler()
        callbacks.append(langfuse_handler)
        logger.info("[Langfuse] Đã kích hoạt LangfuseCallbackHandler để đẩy trace lên server.")
    else:
        logger.info(
            "[Langfuse] Chưa cấu hình LANGFUSE_PUBLIC_KEY / SECRET_KEY -> Chạy local trace log."
        )
    
    run_config: dict[str, Any] = {
        "configurable": {"thread_id": "thread-way-3-demo"},
        "callbacks": callbacks,
        "run_name": "lesson_4_5_create_agent_run",
    }

    # 4. Thực thi Agent (Invoke) với Runtime Context và Email ẩn trong câu hỏi để test PIIMiddleware
    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=run_config,
        context=UserContext(user_id="hoang_lh", user_tier="pro", language="vi"),
    )

    print(f"-> Message cuối cùng (Cách 3): {result['messages'][-1].content}")
    if "structured_response" in result and result["structured_response"] is not None:
        structured: WeatherReport = result["structured_response"]
        print(f"-> Structured Output (WeatherReport): {structured.model_dump_json(indent=2)}")

    return result

if __name__ == "__main__":
    
    sample_query = "Hãy kiểm tra thời tiết tại Hà Nội giúp tôi (email liên hệ: user@example.com)."
    
    # Chạy Cách 1
    run_way1_manual_loop(sample_query)
    
    # Chạy cách 2
    run_way2_langgraph(sample_query)
    
    # Chạy cách 3
    run_way3_create_agent_with_middleware(sample_query)