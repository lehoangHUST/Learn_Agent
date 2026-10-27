import time
import unicodedata
import pytest
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough
)

class DelayedFakeChatModel(FakeListChatModel):
    """Fake model giả lập network latency bằng sleep."""
    delay: float = 0.5

    def _call(self, messages, stop=None, run_manager=None, **kwargs) -> str:
        time.sleep(self.delay)
        return super()._call(messages, stop=stop, run_manager=run_manager, **kwargs)

def clean_input(raw_text: str) -> str:
    text = unicodedata.normalize("NFC", raw_text)
    return " ".join(text.split())

def parse_csv_tags(ai_message) -> list[str]:
    """Parse text đầu ra của model thành list từ khóa sạch."""
    content = ai_message.content if hasattr(ai_message, "content") else str(ai_message)
    return [tag.strip() for tag in content.split(",") if tag.strip()]

def post_process_result(data: dict) -> dict:
    return {
        "summary": data.get("summary", ""),
        "keywords": data.get("keywords", []),
        "metadata": {
            "keyword_count": len(data.get("keywords", [])),
            "summary_word_count": len(data.get("summary", "").split())
        }
    }
    
def build_pipeline(summary_model, keyword_model):
    message_summary = "Tóm tắt nội dung văn bản sau: {text}"
    message_keyword = "Trích xuất các từ khóa cách nhau bằng dấu phẩy: {text}" 
    
    # build promt template
    prompt_summary = ChatPromptTemplate.from_template(message_summary)
    prompt_keyword = ChatPromptTemplate.from_template(message_keyword)
    
    # Branch summary
    branch_summary = prompt_summary | summary_model | StrOutputParser()
    
    # Brach keyword
    branch_keyword = prompt_keyword | keyword_model | RunnableLambda(parse_csv_tags)
    
    # combine pipeline
    pipeline = (
        RunnableLambda(clean_input)
        | {"text": RunnablePassthrough()}
        | RunnableParallel(
            summary=branch_summary,
            keywords=branch_keyword,
        )
        | RunnableLambda(post_process_result)
    )
    
    return pipeline

def test_pipeline_unit_and_parallelism():
    fake_summary_model = DelayedFakeChatModel(
        responses=["Bản tóm tắt nội dung chính."], 
        delay=0.5
    )
    fake_keyword_model = DelayedFakeChatModel(
        responses=["python, langchain, lcel, pipeline"], 
        delay=0.5
    )
    
    pipeline = build_pipeline(fake_summary_model, fake_keyword_model)
    
    print(pipeline)
    raw_input = "   Đây là    nội dung bài báo khoa học  cần được tóm tắt và trích xuất.   "

    # 2. Thực thi và đo thời gian
    start_time = time.perf_counter()
    output = pipeline.invoke(raw_input)
    print(output)
    elapsed_time = time.perf_counter() - start_time

    # 3. Kiểm định Unit Test (Logic & Contract)
    assert output["summary"] == "Bản tóm tắt nội dung chính."
    assert output["keywords"] == ["python", "langchain", "lcel", "pipeline"]
    assert output["metadata"]["keyword_count"] == 4
    assert output["metadata"]["summary_word_count"] == 6

    # 4. Chứng minh song song:
    # Nếu chạy tuần tự: >= 0.5s + 0.5s = 1.0s
    # Chạy song song: ~0.5s (cho phép overhead < 0.85s)
    print(f"\nThời gian thực thi: {elapsed_time:.3f} giây")
    assert elapsed_time < 0.85, (
        f"Pipeline bị nghẽn chạy tuần tự! Tổng thời gian {elapsed_time:.3f}s >= 0.85s"
    )
    assert elapsed_time >= 0.5, "Thời gian thực thi không hợp lệ với mock latency."
    print("✓ Passed: Đã chứng minh hai nhánh thực sự chạy song song!")

if __name__ == "__main__":
    test_pipeline_unit_and_parallelism()
    