import os
import threading
from typing import Dict, Any, List
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langfuse import get_client
from langfuse.langchain import CallbackHandler

# =====================================================================
# 1. CẤU HÌNH MÔI TRƯỜNG
# =====================================================================
os.environ.pop("SSL_CERT_FILE", None)
load_dotenv()

# =====================================================================
# 2. GIẢ LẬP KNOWLEDGE BASE & RETRIEVER (TỪ PHASE 2)
# (Trong project thực tế, bạn thay bằng Qdrant / Chroma / FAISS retriever của Phase 2)
# =====================================================================
KNOWLEDGE_BASE = [
    Document(
        page_content=(
            "Hệ thống AI Gateway bao gồm 2 phiên bản chính: Bản Standard và Bản ProMax. "
            "Bản Standard hỗ trợ định tuyến cơ bản (Round-Robin) với giới hạn 100 requests/phút, "
            "phù hợp cho nhóm phát triển nhỏ."
        ),
        metadata={"source": "doc_overview.md", "id": "doc_1"},
    ),
    Document(
        page_content=(
            "Chi tiết về Bản Standard: Chi phí là 29 USD/tháng. Không hỗ trợ Semantic Caching "
            "và không có tính năng tự động Fallback khi nhà cung cấp LLM bị sập."
        ),
        metadata={"source": "doc_standard.md", "id": "doc_2"},
    ),
    Document(
        page_content=(
            "Chi tiết về Bản ProMax: Chi phí là 199 USD/tháng. Hỗ trợ tới 5.000 requests/phút, "
            "tích hợp sẵn Redis Semantic Caching, tự động Fallback đa đám mây và ẩn dữ liệu nhạy cảm (PII Masking)."
        ),
        metadata={"source": "doc_promax.md", "id": "doc_3"},
    ),
]


def simple_keyword_retriever(query: str) -> List[Document]:
    """
    Hàm Retriever minh họa cơ chế khớp từ khóa/ngữ nghĩa.
    Nếu câu hỏi chỉ là 'Còn cái thứ hai giá bao nhiêu?' (thiếu chữ 'ProMax'),
    retriever sẽ KHÔNG tìm thấy doc_3.
    """
    query_lower = query.lower()
    results = []

    if "hai phiên bản" in query_lower or "2 phiên bản" in query_lower or "ai gateway" in query_lower:
        results.append(KNOWLEDGE_BASE[0])
    if "standard" in query_lower:
        results.append(KNOWLEDGE_BASE[1])
    if "promax" in query_lower or "pro max" in query_lower:
        results.append(KNOWLEDGE_BASE[2])

    # Fallback nếu không khớp từ khóa cụ thể nào (trả về doc tổng quan)
    if not results:
        results.append(KNOWLEDGE_BASE[0])

    return results


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[{doc.metadata['source']}]: {doc.page_content}" for doc in docs
    )


retriever_runnable = RunnableLambda(simple_keyword_retriever).with_config(
    run_name="Phase2_Vector_Retriever"
)

# =====================================================================
# 3. KHỞI TẠO 2 MODEL: MODEL NHỎ (REWRITER) & MODEL CHÍNH (GENERATOR)
# =====================================================================
# Dùng model nhỏ, tốc độ cao, chi phí rẻ cho bước viết lại câu hỏi
rewriter_llm = init_chat_model(
    "groq:allam-2-7b",  # Hoặc "groq:openai/gpt-oss-20b" nếu muốn dùng chung
    temperature=0,
).with_config(run_name="Rewriter_Small_LLM")

# Model chính chịu trách nhiệm suy luận và trả lời dựa trên Context
main_llm = init_chat_model(
    "groq:openai/gpt-oss-20b",
    temperature=0,
).with_config(run_name="Main_Generator_LLM")

# =====================================================================
# 4. THIẾT KẾ CONTEXTUAL QUESTION REWRITING CHAIN
# =====================================================================
contextualize_system_prompt = (
    "Dựa vào lịch sử trò chuyện (history) và câu hỏi mới nhất của người dùng (input), "
    "nhiệm vụ của bạn là viết lại câu hỏi mới nhất thành một câu hỏi độc lập (standalone question) "
    "có thể hiểu được đầy đủ mà KHÔNG cần đọc lịch sử trò chuyện.\n"
    "Quy tắc bắt buộc:\n"
    "1. Thay thế toàn bộ các đại từ chỉ định mơ hồ (như 'nó', 'cái đó', 'cái thứ hai', 'gói vừa rồi') "
    "bằng danh từ/tên riêng cụ thể đã nhắc đến trong lịch sử.\n"
    "2. TUYỆT ĐỐI KHÔNG trả lời câu hỏi, chỉ trả về đúng câu hỏi đã được viết lại.\n"
    "3. Nếu câu hỏi mới nhất đã rõ nghĩa và độc lập, hãy giữ nguyên nó."
)

contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# Chain thực hiện viết lại câu hỏi (chỉ kích hoạt khi history có dữ liệu)
rewrite_chain = (
    contextualize_prompt
    | rewriter_llm
    | StrOutputParser()
).with_config(run_name="Question_Rewriting_Step")


def has_chat_history(x: Dict[str, Any]) -> bool:
    """Kiểm tra xem phiên hiện tại đã có lịch sử hay chưa."""
    history = x.get("history", [])
    return len(history) > 0


# Rẽ nhánh:
# - Nếu ĐÃ CÓ lịch sử -> chạy `rewrite_chain` (tốn 1 LLM call nhỏ)
# - Nếu CHƯA CÓ lịch sử (Lượt 1 hoặc chạy Eval Phase 2) -> giữ nguyên `input` (0 LLM call)
contextualized_question_branch = RunnableBranch(
    (
        has_chat_history,
        rewrite_chain,
    ),
    RunnableLambda(lambda x: x["input"]).with_config(
        run_name="Bypass_Rewrite_Turn_1"
    ),
).with_config(run_name="Conditional_Contextualizer")


# Hàm bọc lại bước lấy standalone_query để đẩy cả câu hỏi gốc và câu hỏi viết lại
# vào một dictionary phục vụ hiển thị cạnh nhau trên Langfuse Trace
def build_queries_inspection(x: Dict[str, Any]) -> Dict[str, Any]:
    standalone_q = contextualized_question_branch.invoke(x)
    return {
        "original_question": x["input"],
        "rewritten_question": standalone_q,
        "is_rewritten": len(x.get("history", [])) > 0,
    }


query_inspector = RunnableLambda(build_queries_inspection).with_config(
    run_name="Query_Comparison_Inspector"
)

# =====================================================================
# 5. LẮP RÁP RAG PIPELINE HOÀN CHỈNH
# =====================================================================
qa_system_prompt = (
    "Bạn là trợ lý kỹ thuật AI. Hãy sử dụng ngữ cảnh (Context) được cung cấp dưới đây "
    "để trả lời câu hỏi của người dùng một cách chính xác, ngắn gọn.\n"
    "Nếu thông tin không có trong Context, hãy nói rằng bạn không biết, tuyệt đối không tự bịa.\n\n"
    "--- CONTEXT ---\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# Luồng dữ liệu chuẩn:
# 1. Tạo `query_info` chứa `original_question` và `rewritten_question` cạnh nhau.
# 2. Đi retrieve tài liệu bằng `rewritten_question`.
# 3. Đưa `context`, `history` và câu hỏi gốc `input` vào `qa_prompt` để sinh câu trả lời.
rag_pipeline = (
    RunnablePassthrough.assign(query_info=query_inspector)
    | RunnablePassthrough.assign(
        retrieved_docs=lambda x: retriever_runnable.invoke(
            x["query_info"]["rewritten_question"]
        )
    )
    | RunnablePassthrough.assign(
        context=lambda x: format_docs(x["retrieved_docs"])
    )
    | RunnablePassthrough.assign(
        answer=(qa_prompt | main_llm | StrOutputParser()).with_config(
            run_name="Final_Answer_Generation"
        )
    )
).with_config(run_name="Conversational_RAG_Pipeline")


# =====================================================================
# 6. QUẢN LÝ LỊCH SỬ SESSION & TÍCH HỢP LANGFUSE
# (Bạn có thể cắm trực tiếp class SecureSQLiteChatMessageHistory từ lesson3_2 vào đây)
# =====================================================================
store: Dict[str, InMemoryChatMessageHistory] = {}
_store_lock = threading.Lock()


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    with _store_lock:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]


conversational_rag = RunnableWithMessageHistory(
    runnable=rag_pipeline,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",  # Chỉ định rõ key chứa câu trả lời để lưu vào history
)


def ask_rag(user_id: str, session_id: str, user_input: str) -> Dict[str, Any]:
    """Thực thi Conversational RAG và đẩy metadata chi tiết lên Langfuse."""
    langfuse_handler = CallbackHandler()

    config = {
        "configurable": {"session_id": session_id},
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_user_id": user_id,
            "langfuse_session_id": session_id,
            "langfuse_tags": ["lesson-3.3", "contextual-rag"],
        },
    }

    try:
        result = conversational_rag.invoke(
            {"input": user_input},
            config=config,
        )
    finally:
        get_client().flush()

    return result


# =====================================================================
# 7. BỘ TEST CHỨNG MINH PASS CÁC TIÊU CHÍ (ACCEPTANCE TESTS)
# =====================================================================
if __name__ == "__main__":
    user_id = "user_hoang"
    session_id = "sess_rag_demo_01"

    print("=================================================================")
    print("TIÊU CHÍ 1: LƯỢT ĐẦU TIÊN (TURN 1) - KHÔNG TỐN THÊM LLM CALL")
    print("=================================================================")
    q1 = "Hệ thống AI Gateway có mấy phiên bản chính? Hãy kể tên chúng."
    res1 = ask_rag(user_id, session_id, q1)

    print(f"User Question      : {res1['query_info']['original_question']}")
    print(f"Rewritten Question : {res1['query_info']['rewritten_question']}")
    print(f"Is Rewritten?      : {res1['query_info']['is_rewritten']} (Bypass LLM Rewriter!)")
    print(f"Retrieved Docs     : {[d.metadata['source'] for d in res1['retrieved_docs']]}")
    print(f"AI Answer          : {res1['answer']}\n")

    # Kiểm chứng tự động Tiêu chí 1
    assert res1["query_info"]["is_rewritten"] is False, "Lỗi: Lượt 1 vẫn gọi LLM Rewriter!"
    assert res1["query_info"]["original_question"] == res1["query_info"]["rewritten_question"]

    print("=================================================================")
    print("TIÊU CHÍ 2: CÂU HỎI TIẾP NỐI (TURN 2) - RETRIEVE ĐÚNG TÀI LIỆU")
    print("=================================================================")
    # Câu hỏi mơ hồ: dùng cụm 'cái thứ hai' (ám chỉ bản ProMax đã nhắc ở Turn 1)
    q2 = "Còn cái thứ hai thì chi phí bao nhiêu và có những tính năng gì đặc biệt?"
    res2 = ask_rag(user_id, session_id, q2)

    print(f"Original Question  : {res2['query_info']['original_question']}")
    print(f"Rewritten Question : {res2['query_info']['rewritten_question']}")
    print(f"Is Rewritten?      : {res2['query_info']['is_rewritten']}")
    print(f"Retrieved Docs     : {[d.metadata['source'] for d in res2['retrieved_docs']]}")
    print(f"AI Answer          : {res2['answer']}\n")

    # Kiểm chứng tự động Tiêu chí 2: Phải lấy được doc_promax.md và biết giá 199 USD
    retrieved_sources = [d.metadata["source"] for d in res2["retrieved_docs"]]
    assert "doc_promax.md" in retrieved_sources, "Lỗi: Không retrieve được tài liệu của bản ProMax!"
    assert "199" in res2["answer"], "Lỗi: Câu trả lời không chứa đúng giá của bản ProMax!"

    print("=================================================================")
    print("TIÊU CHÍ 3: BỘ EVAL CỦA PHASE 2 (STATELESS) KHÔNG BỊ ẢNH HƯỞNG")
    print("=================================================================")
    # Khi chạy benchmark Phase 2, mỗi câu hỏi chạy trên một session_id mới (history rỗng)
    eval_session = "eval_phase2_stateless"
    q_eval = "Chi phí của bản Standard là bao nhiêu mỗi tháng?"
    res_eval = ask_rag(user_id, eval_session, q_eval)

    print(f"Eval Question      : {res_eval['query_info']['original_question']}")
    print(f"Rewritten Question : {res_eval['query_info']['rewritten_question']}")
    print(f"Retrieved Docs     : {[d.metadata['source'] for d in res_eval['retrieved_docs']]}")
    print(f"AI Answer          : {res_eval['answer']}\n")

    assert res_eval["query_info"]["is_rewritten"] is False
    assert "doc_standard.md" in [d.metadata["source"] for d in res_eval["retrieved_docs"]]
    print("✅ TẤT CẢ CÁC BÀI TEST ĐỀU PASS TUYỆT ĐỐI!")