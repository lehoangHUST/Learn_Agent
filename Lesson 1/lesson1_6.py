"""
Lesson 1.5 & 1.6: Resilient Pipeline & Safe Langfuse Observability
- Sử dụng Groq (ChatGroq) miễn phí, tốc độ cao.
- Đảm bảo cơ chế Graceful Degradation khi thiếu Langfuse keys.
- Cấu hình RunnableConfig chuẩn, bảo mật metadata không chứa secret.
"""

import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
import asyncio
import time
from dotenv import load_dotenv
from typing import List, Optional, Tuple, Type
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import Runnable, RunnableConfig
from groq import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError, NotFoundError

load_dotenv()

# =====================================================================
# 0. KIỂM TRA MÔI TRƯỜNG & KHỞI TẠO AN TOÀN LANGFUSE HANDLER
# =====================================================================

if not os.environ.get("GROQ_API_KEY"):
    print("⚠️  Chưa có GROQ_API_KEY. Hãy thiết lập: export GROQ_API_KEY='gsk_...'")

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


# =====================================================================
# 1. XÂY DỰNG PIPELINE CHỊU LÕI (RETRY + FALLBACK + RATE LIMITER)
# =====================================================================

def get_models_and_exceptions(
    rate_limiter: InMemoryRateLimiter,
) -> Tuple[BaseChatModel, BaseChatModel, Tuple[Type[Exception], ...]]:

    # Các lỗi tạm thời (Transient Errors) từ phía Groq API
    transient_exceptions = (
        RateLimitError,       # HTTP 429 (Groq Free Tier rất dễ chạm giới hạn RPM/TPM nếu gửi dồn dập)
        APITimeoutError,      # Request Timeout
        APIConnectionError,   # Lỗi kết nối mạng
        InternalServerError,  # HTTP 5xx
        NotFoundError
    )

    # Model chính (Chất lượng cao)
    primary = init_chat_model(
        "groq:openai/gpt-oss-20b",
        temperature=0,
        rate_limiter=rate_limiter
    )

    # Model dự phòng (Nhỏ, cực nhanh, hạn mức rate limit rộng hơn)
    fallback = init_chat_model(
        "groq:allam-2-7b",
        temperature=0,
        rate_limiter=rate_limiter
    )

    return primary, fallback, transient_exceptions

def build_resilient_pipeline() -> Runnable:
    # Client-side Rate Limiter giúp làm mượt request, tránh 429 trên Groq Free Tier
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=2.0,
        check_every_n_seconds=0.1,
        max_bucket_size=5,
    )

    transient_exceptions = (
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
        InternalServerError,
    )

    # Model chính (Llama 3.3 70B - chất lượng cao) kèm Retry có giới hạn
    
    base_primary_llm, fallback_llm, transient_exceptions = get_models_and_exceptions(rate_limiter)
        
    primary_with_retry = base_primary_llm.with_retry(
        retry_if_exception_type=transient_exceptions,
        wait_exponential_jitter=True,  # Exponential backoff + Jitter tránh thundering herd
        stop_after_attempt=3,          # Thử tối đa 3 lần (1 lần gốc + 2 lần retry)
    )

    resilient_llm = primary_with_retry.with_fallbacks(
        fallbacks=[fallback_llm],
        exceptions_to_handle=transient_exceptions,
    )

    # LCEL Pipeline: Prompt -> LLM -> Parser (Langfuse tự động tạo observation tree cho từng bước)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Bạn là một chuyên gia công nghệ. Hãy giải thích ngắn gọn bằng tiếng Việt trong tối đa 3 câu."),
        ("human", "Giải thích khái niệm: {topic}"),
    ])
    parser = StrOutputParser()

    return prompt | resilient_llm | parser


# =====================================================================
# 2. TẠO RUNNABLE CONFIG CHUẨN & BẢO MẬT METADATA
# =====================================================================

def create_execution_config(run_name: str, max_concurrency: int = 3, user_id: str = "user_safe_01") -> RunnableConfig:
    """
    Tạo config chuẩn chứa:
    - run_name rõ ràng cho Trace.
    - Metadata an toàn (KHÔNG chứa API key, mật khẩu hay PII nhạy cảm).
    - Callback tự động xử lý an toàn khi thiếu key Langfuse.
    """
    safe_callbacks = get_safe_langfuse_callbacks()

    return RunnableConfig(
        run_name=run_name,
        tags=["lesson-1.5", "lesson-1.6", "production-ready"],
        metadata={
            "user_id": user_id,          # ID người dùng ẩn danh / an toàn
            "environment": "staging",    # Môi trường chạy
            "model_primary": "llama-3.3-70b-versatile",
        },
        callbacks=safe_callbacks,
        max_concurrency=max_concurrency,
    )


# =====================================================================
# 3. STREAMING OUTPUT RA TERMINAL
# =====================================================================

def demo_streaming(pipeline: Runnable, topic: str) -> None:
    print("=" * 70)
    print(f"🚀 STREAMING OUTPUT RA TERMINAL (Topic: '{topic}')")
    print("=" * 70)

    config = create_execution_config(run_name="stream_single_topic_run")
    print("👉 Output: ", end="", flush=True)

    start_time = time.perf_counter()
    first_token_time = None

    for chunk in pipeline.stream({"topic": topic}, config=config):
        if first_token_time is None:
            first_token_time = time.perf_counter() - start_time
        print(chunk, end="", flush=True)

    total_time = time.perf_counter() - start_time
    print(f"\n\n⏱️  Time to First Token (TTFT): {first_token_time:.3f}s | Tổng thời gian: {total_time:.3f}s\n")


# =====================================================================
# 4. BATCH 10 TOPICS, GIỚI HẠN CONCURRENCY & AN TOÀN LỖI
# =====================================================================

def demo_batch_processing(pipeline: Runnable, topics: List[str], max_concurrency: int = 3) -> None:
    print("=" * 70)
    print(f"🚀 BATCH {len(topics)} TOPICS VỚI RETURN_EXCEPTIONS=True (max_concurrency={max_concurrency})")
    print("=" * 70)

    inputs = [{"topic": t} for t in topics]
    config = create_execution_config(run_name="batch_10_topics_run", max_concurrency=max_concurrency)

    print(f"Đang thực thi pipeline.batch()...")
    t0 = time.perf_counter()
    
    # return_exceptions=True đảm bảo 1 phần tử lỗi không làm mất kết quả của các phần tử khác trong lô
    results = pipeline.batch(inputs, config=config, return_exceptions=True)
    
    elapsed_batch = time.perf_counter() - t0
    print(f"✅ Hoàn thành Batch trong: {elapsed_batch:.2f} giây.\n")

    for idx, (topic, res) in enumerate(zip(topics, results), start=1):
        if isinstance(res, Exception):
            print(f"  ❌ [{idx:02d}] {topic}: Gặp lỗi -> {type(res).__name__}: {res}")
        else:
            short_res = res.replace("\n", " ")[:80]
            print(f"  ✔️  [{idx:02d}] {topic}: {short_res}...")


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def main() -> None:
    pipeline = build_resilient_pipeline()

    topics_10 = [
        "Retrieval-Augmented Generation (RAG)",
        "Vector Database & Embeddings",
        "LangChain Expression Language (LCEL)",
        "Exponential Backoff và Jitter",
        "Token Streaming trong LLM",
        "Function Calling / Tool Calling",
        "Prompt Injection & Guardrails",
        "Semantic Caching cho LLM",
        "Rate Limiting phía Client",
        "Observability với Langfuse",
    ]

    # 1. Chạy Stream
    demo_streaming(pipeline, topic="LangChain Expression Language (LCEL)")

    # 2. Chạy Batch 10 topics
    demo_batch_processing(pipeline, topics=topics_10, max_concurrency=3)

    # 3. Đồng bộ dữ liệu trace lên Langfuse trước khi thoát (nếu có dùng handler)
    # Langfuse v3+: CallbackHandler không còn .flush(), phải flush qua client.
    if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
        try:
            from langfuse import get_client
            print("\n🔄 Đang đồng bộ trace dữ liệu lên Langfuse Cloud...")
            get_client(public_key=os.environ["LANGFUSE_PUBLIC_KEY"]).flush()
            print("✨ Hoàn tất đồng bộ trace!")
        except Exception as e:
            print(f"⚠️ Không flush được Langfuse: {e}")


if __name__ == "__main__":
    main()