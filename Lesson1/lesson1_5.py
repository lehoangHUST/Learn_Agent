"""
Lesson 1.5: Execution, Configuration & Resilience Pipeline (Phiên bản Groq / xAI Grok)
Mục tiêu:
- Chạy pipeline hiệu quả (Batching, Streaming, Async, Rate Limiting).
- Truyền cấu hình chuẩn qua RunnableConfig (tags, metadata, callbacks, run_name, max_concurrency).
- Chịu lỗi tạm thời từ phía provider (with_retry, with_fallbacks).
- Sử dụng Groq (GROQ_API_KEY - miễn phí, tốc độ cao) hoặc xAI Grok (XAI_API_KEY) thay vì OpenAI.
"""
import certifi
import os
os.environ["SSL_CERT_FILE"] = certifi.where()

import asyncio
import time
from typing import List, Tuple, Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import Runnable, RunnableConfig
from groq import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError, NotFoundError

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


def create_execution_config(
    run_name: str,
    max_concurrency: int = 3,
    user_id: str = "user_demo_01"
) -> RunnableConfig:
    return RunnableConfig(
        run_name=run_name,
        tags=["lesson-1.5", "groq-pipeline", "resilience"],
        metadata={
            "user_id": user_id,
            "lesson": "1.5",
            "provider": "groq",
        },
        callbacks=[],
        max_concurrency=max_concurrency
    )

# init pipeline
def build_pipeline() -> Runnable:
    
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=2,
        check_every_n_seconds=0.1,
        max_bucket_size=5
    )
    
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
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Bạn là một chuyên gia công nghệ. Hãy giải thích ngắn gọn, súc tích bằng tiếng Việt trong tối đa 8 câu."),
        ("human", "Giải thích khái niệm: {topic}")
    ])
    
    parser = StrOutputParser()
    
    pipeline = prompt | resilient_llm | parser
    return pipeline

# Run demo async
def demo_sync(pipeline: Runnable, topic: str) -> None:
    
    config = create_execution_config(run_name="demo_run")
    start_time = time.perf_counter()
    
    response = pipeline.invoke([
        {"topic": topic}
    ])
    elapsed_time = time.perf_counter() - start_time
    print("Check response: ", response)
    print(f"\nThời gian thực thi: {elapsed_time:.3f} giây")
    
def demo_streaming(pipeline: Runnable, topic: str) -> None:
    config = create_execution_config(run_name="demo_streaming")
    
    start_time = time.perf_counter()
    first_token_time = None
    
    for chunk in pipeline.stream({"topic": topic}, config=config):
        if first_token_time is None:
            first_token_time = time.perf_counter() - start_time
        print(chunk, end="", flush=True)

    total_time = time.perf_counter() - start_time
    ttft_display = f"{first_token_time:.3f}s" if first_token_time is not None else "N/A"
    print(f"\n\n⏱️  Time to First Token (TTFT): {ttft_display} | Tổng thời gian: {total_time:.3f}s")


def demo_batch_processing(pipeline: Runnable, topics: List[str], max_concurrency: int = 3) -> None:
    """
    Minh họa:
    - Chạy batch 10 topics với config={"max_concurrency": n}
    - Sử dụng return_exceptions=True để 1 input lỗi không làm hỏng cả lô
    - Sử dụng batch_as_completed để nhận kết quả ngay khi từng task hoàn thành
    """
    print("\n" + "=" * 70)
    print(f"🚀 BÀI TẬP 2: BATCH {len(topics)} TOPICS (max_concurrency={max_concurrency})")
    print("=" * 70)

    inputs = [{"topic": t} for t in topics]
    config = create_execution_config(
        run_name="batch_10_topics_run",
        max_concurrency=max_concurrency,
    )

    # -----------------------------------------------------------------
    # Cách 1: .batch() tiêu chuẩn (Trả về đúng thứ tự input ban đầu)
    # -----------------------------------------------------------------
    print(f"\n[1] Đang chạy pipeline.batch() với max_concurrency={max_concurrency}, return_exceptions=True...")
    t0 = time.perf_counter()

    results = pipeline.batch(
        inputs,
        config=config,
        return_exceptions=True,  # Rất quan trọng: giữ cho cả lô không bị crash nếu 1 request lỗi
    )

    elapsed_batch = time.perf_counter() - t0
    print(f"✅ Hoàn thành .batch() {len(topics)} topics trong: {elapsed_batch:.2f} giây.\n")

    for idx, (topic, res) in enumerate(zip(topics, results), start=1):
        if isinstance(res, Exception):
            print(f"  ❌ [{idx:02d}] {topic}: LỖI -> {type(res).__name__}: {res}")
        else:
            short_res = res.replace("\n", " ")[:95]
            print(f"  ✔️  [{idx:02d}] {topic}: {short_res}...")

    print(f"\n[2] Đang chạy pipeline.batch_as_completed() (nhận kết quả ngay khi xong)...")
    t1 = time.perf_counter()

    for completed_idx, result in pipeline.batch_as_completed(
        inputs,
        config=config,
        return_exceptions=True,
    ):
        finish_time = time.perf_counter() - t1
        topic_name = topics[completed_idx]
        if isinstance(result, Exception):
            print(f"  ⏱️  +{finish_time:.2f}s | ❌ Input #{completed_idx} ({topic_name}) lỗi: {result}")
        else:
            short_res = result.replace("\n", " ")[:75]
            print(f"  ⏱️  +{finish_time:.2f}s | ✔️  Input #{completed_idx:02d} ({topic_name}) xong: {short_res}...")

    elapsed_as_completed = time.perf_counter() - t1
    print(f"✅ Tổng thời gian .batch_as_completed(): {elapsed_as_completed:.2f} giây.")

async def demo_async_batch(pipeline: Runnable, topics: List[str], max_concurrency: int = 5) -> None:
    """
    Minh họa abatch() khi ứng dụng chạy trên nền tảng bất đồng bộ (FastAPI, Async Worker)
    và có nhiều request I/O đồng thời.
    """
    print("\n" + "=" * 70)
    print(f"⚡ MINH HỌA THÊM: ASYNC ABATCH ({len(topics)} topics, max_concurrency={max_concurrency})")
    print("=" * 70)

    inputs = [{"topic": t} for t in topics]
    config = create_execution_config(
        run_name="async_abatch_10_topics",
        max_concurrency=max_concurrency,
    )

    t0 = time.perf_counter()
    results = await pipeline.abatch(
        inputs,
        config=config,
        return_exceptions=True,
    )
    print("Check results: ", results)
    elapsed = time.perf_counter() - t0

    success_count = sum(1 for r in results if not isinstance(r, Exception))
    print(f"✅ Hoàn thành await pipeline.abatch() ({success_count}/{len(topics)} thành công) trong: {elapsed:.2f} giây.")

    
# ==========================================
# MAIN EXECUTION
# ==========================================

async def main() -> None:
    # init pipeline + retry + fallback + rate limiter with groq
    pipeline = build_pipeline()
    
    topics_10 = [
        "Retrieval-Augmented Generation (RAG)",
        "Vector Database & Embeddings",
        "LangChain Expression Language (LCEL)",
        "Exponential Backoff và Jitter",
        "Token Streaming trong LLM",
        "Function Calling / Tool Calling",
        "Prompt Injection & Guardrails",
        "Semantic Caching cho LLM",
        "Rate Limiting (Client-side vs Provider-side)",
        "Observability & Tracing với Langfuse",
    ]
    
    # Demo sync
    demo_sync(pipeline, topic=topics_10[5])
    
    # Demo streaming
    demo_streaming(pipeline, topic=topics_10[5])
    
    # Demo batch processing
    demo_batch_processing(pipeline, topics=topics_10, max_concurrency=5)
    
    # Demo async streaming
    await demo_async_batch(pipeline, topics=topics_10, max_concurrency=5)
    

if __name__ == "__main__":
    asyncio.run(main())