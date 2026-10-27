import time
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langfuse import get_client, observe

# Load biến môi trường từ file .env (nếu có)
os.environ.pop("SSL_CERT_FILE", None)
load_dotenv()

# =====================================================================
# 0. KHỞI TẠO LANGFUSE CLIENT & CHAT MODEL (GROQ)
# =====================================================================
# Đặt tên biến là lf_client để không bao giờ đụng độ với `import langfuse`
lf_client = get_client()

chat = init_chat_model("groq:openai/gpt-oss-20b", temperature=0.0)


# =====================================================================
# 1. ĐỊNH NGHĨA TOOLS
# =====================================================================

@tool
@observe(as_type="tool", name="get_stock_price")
def get_stock_price(ticker: str) -> str:
    """Tra cứu giá cổ phiếu hoặc chứng chỉ quỹ theo mã ticker (VD: FPT, DCDS, VESAF)."""
    mock_db = {
        "FPT": 135000,
        "VCB": 92000,
        "DCDS": 82500,
        "VESAF": 31200,
    }
    ticker_upper = ticker.upper().strip()
    if ticker_upper not in mock_db:
        raise ValueError(
            f"Không tìm thấy mã '{ticker}'. Các mã hỗ trợ: {list(mock_db.keys())}"
        )
    return f"Giá của {ticker_upper} là {mock_db[ticker_upper]} VND"


@tool
@observe(as_type="tool", name="calculate_investment")
def calculate_investment(price_vnd: float, quantity: int) -> str:
    """Tính tổng số tiền cần đầu tư bằng giá một đơn vị nhân với số lượng."""
    if quantity <= 0:
        raise ValueError("Số lượng mua (quantity) phải lớn hơn 0.")
    total = price_vnd * quantity
    return f"Tổng số tiền đầu tư: {total:,.0f} VND"


TOOLS = [get_stock_price, calculate_investment]
TOOL_REGISTRY: Dict[str, Any] = {t.name: t for t in TOOLS}

chat_with_tools = chat.bind_tools(TOOLS)


# =====================================================================
# 2. HÀM GỌI MODEL (.invoke) & THỰC THI TOOL AN TOÀN
# =====================================================================

@observe(as_type="generation", name="call_llm")
def call_llm(messages: List[BaseMessage]) -> tuple[AIMessage, int]:
    """
    Gọi chat_with_tools.invoke() và cập nhật thông tin Generation lên Langfuse v3.
    """
    response: AIMessage = chat_with_tools.invoke(messages)

    usage = getattr(response, "usage_metadata", None) or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

    # Dùng lf_client (instance của Langfuse) để cập nhật generation hiện tại
    lf_client.update_current_generation(
        input=[{"type": m.type, "content": m.content} for m in messages],
        output={
            "content": response.content,
            "tool_calls": response.tool_calls,
        },
        model="openai/gpt-oss-20b",
        usage_details={
            "input": input_tokens,
            "output": output_tokens,
            "total": total_tokens,
        },
    )

    return response, total_tokens


def execute_tool_safely(tool_call: Dict[str, Any]) -> str:
    """Bắt mọi lỗi khi gọi tool và trả về dưới dạng chuỗi Observation."""
    t_name = tool_call.get("name", "")
    t_args = tool_call.get("args", {})

    if t_name not in TOOL_REGISTRY:
        return f"Error: Tool '{t_name}' không tồn tại. Các tool hợp lệ: {list(TOOL_REGISTRY.keys())}"

    if not isinstance(t_args, dict):
        return f"Error: Arguments cho tool '{t_name}' phải là một dict, nhận được: {t_args}"

    try:
        selected_tool = TOOL_REGISTRY[t_name]
        result = selected_tool.invoke(t_args)
        return str(result)
    except Exception as e:
        return f"Error khi thực thi tool '{t_name}' với args {t_args}: {str(e)}"


def trim_history_safely(
    messages: List[BaseMessage], max_messages: int = 12
) -> List[BaseMessage]:
    """Cắt bớt lịch sử nếu quá dài nhưng không làm đứt rời ToolMessage."""
    if len(messages) <= max_messages:
        return messages

    head = messages[:2]
    tail = messages[-(max_messages - 2) :]

    while tail and isinstance(tail[0], ToolMessage):
        tail.pop(0)

    return head + tail


# =====================================================================
# 3. VÒNG LẶP AGENT BẰNG PYTHON THUẦN
# =====================================================================

@observe(as_type="agent", name="run_agent_loop")
def run_agent_loop(
    user_query: str,
    max_iterations: int = 5,
    max_token_budget: int = 6000,
    timeout_seconds: float = 30.0,
) -> str:
    start_time = time.time()
    total_tokens_used = 0

    messages: List[BaseMessage] = [
        SystemMessage(
            content=(
                "Bạn là trợ lý tài chính thông minh. Hãy dùng các tool được cung cấp để lấy số liệu "
                "và tính toán chính xác. Nếu một tool trả về Error, hãy đọc thông báo lỗi và thông báo "
                "rõ cho người dùng thay vì tự bịa số liệu."
            )
        ),
        HumanMessage(content=user_query),
    ]

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- [Vòng {iteration}/{max_iterations}] ---")

        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            stop_msg = f"[Dừng] Quá thời gian ({elapsed:.1f}s > {timeout_seconds}s)."
            print(stop_msg)
            return stop_msg

        messages = trim_history_safely(messages, max_messages=14)

        # Bước 1: Gọi Model
        ai_msg, step_tokens = call_llm(messages)
        total_tokens_used += step_tokens
        print(
            f"Token bước này: {step_tokens} | Tổng token: {total_tokens_used}/{max_token_budget}"
        )

        messages.append(ai_msg)

        # Điều kiện dừng: Không còn tool_calls
        if not ai_msg.tool_calls:
            print("-> Hoàn tất! Model không gọi thêm tool.")
            return str(ai_msg.content)

        # Bước 2 & 3: Thực thi tool và thêm ToolMessage
        for tool_call in ai_msg.tool_calls:
            t_name = tool_call["name"]
            t_args = tool_call["args"]
            t_id = tool_call["id"]

            print(f"-> Gọi Tool: {t_name}({t_args})")
            observation = execute_tool_safely(tool_call)
            print(f"<- Observation: {observation}")

            messages.append(
                ToolMessage(
                    content=observation,
                    tool_call_id=t_id,
                    name=t_name,
                )
            )

        if total_tokens_used >= max_token_budget:
            stop_msg = f"[Dừng] Vượt ngân sách token ({total_tokens_used} >= {max_token_budget})."
            print(stop_msg)
            return stop_msg

    stop_msg = f"[Dừng] Đạt giới hạn số vòng lặp tối đa (max_iterations={max_iterations})."
    print(stop_msg)
    return stop_msg


# =====================================================================
# 4. CHẠY KIỂM THỬ
# =====================================================================

if __name__ == "__main__":
    question = (
        "Kiểm tra giúp tôi giá của mã FPT. "
        "Sau đó tính tổng số tiền nếu tôi mua 200 cổ phiếu FPT."
    )

    print(f"Câu hỏi: {question}")
    answer = run_agent_loop(
        user_query=question,
        max_iterations=5,
        max_token_budget=6000,
        timeout_seconds=30.0,
    )

    print("\n======================================")
    print("CÂU TRẢ LỜI CUỐI CÙNG:")
    print(answer)

    # Flush qua lf_client
    lf_client.flush()