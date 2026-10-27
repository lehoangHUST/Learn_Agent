import os
from dotenv import load_dotenv
import httpx
import groq
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

# Khắc phục lỗi cert đường dẫn ảo nếu có
os.environ.pop("SSL_CERT_FILE", None)
load_dotenv()

config = RunnableConfig(
    run_name="Explain_Topic_Pipeline",
    tags=["demo-pipeline", "lesson-lcel"],
    metadata={"user_id": "user_123", "environment": "production"},
    # callbacks=[...] # Gắn handler như Langfuse vào đây, mọi bước con đều nhận được
)

# ==========================================
# 1. KIỂM TRA LỖI THIẾU API KEY
# ==========================================
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError(
        "[LỖI THIẾU KEY] Chưa tìm thấy biến môi trường 'GROQ_API_KEY'. "
        "Vui lòng khai báo trong file .env hoặc export vào terminal."
    )
    
try:
    # Khởi tạo Chat Model với timeout cụ thể (ví dụ 10s)
    # Lưu ý: Thay tên model hợp lệ trên Groq như 'llama-3.1-8b-instant' hoặc 'llama-3.3-70b-versatile'
    chat = init_chat_model("groq:allam-2-7b")

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of France?"),
    ]

    response = chat.invoke(messages)
    
    print("Nội dung phản hồi:\n", response.content)
    print("\nUsage Metadata:\n", response.usage_metadata)
    print("\nResponse Metadata:\n", response.response_metadata)

# ==========================================
# 2. XỬ LÝ LỖI TIMEOUT VÀ LỖI PROVIDER RIÊNG BIỆT
# ==========================================
except (httpx.TimeoutException, groq.APITimeoutError) as e:
    print(f"[LỖI TIMEOUT] Request đến Groq quá thời gian chờ quy định (Timeout): {e}")

except groq.NotFoundError as e:
    print(f"[LỖI PROVIDER - 404] Model không tồn tại hoặc tài khoản không có quyền truy cập: {e}")

except groq.AuthenticationError as e:
    print(f"[LỖI PROVIDER - 401] Xác thực thất bại, Groq API Key không chính xác hoặc đã bị thu hồi: {e}")

except groq.RateLimitError as e:
    print(f"[LỖI PROVIDER - 429] Vượt quá giới hạn gọi API (Rate limit / Quota exceeded): {e}")

except groq.APIStatusError as e:
    print(f"[LỖI PROVIDER - HTTP {e.status_code}] Nhà cung cấp Groq trả về lỗi máy chủ: {e.message}")

except Exception as e:
    print(f"[LỖI HỆ THỐNG KHÁC] Ngoại lệ không xác định: {e}")
    
    
"""
1. LLM (Text completion) -> Nhận vào là chuỗi văn bản thuần, sau đó trả về chuỗi văn bản dạng nối tiếp
2. Chat Model:  Nhận vào danh sách các tin nhắn có gắn nhãn vai trò (List[BaseMessage] gồm SystemMessage, HumanMessage, AIMessage, ToolMessage) và trả về một đối tượng AIMessage.Cơ chế: Được tinh chỉnh chuyên biệt (Fine-tuned / RLHF) theo cấu trúc hội thoại đa lượt (multi-turn conversation) và hỗ trợ gọi công cụ (tool calling / function calling).Ví dụ: Input: [SystemMessage(...), HumanMessage(...)] $\rightarrow$ Output: AIMessage(...). Các model hiện đại như GPT-4o, Claude 3.5, Llama 3.3.
"""