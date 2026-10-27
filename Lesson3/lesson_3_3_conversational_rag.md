# Lesson 3.3 — Conversational RAG

## 1. Mục tiêu

Thêm bước **contextual question rewriting** vào RAG của Phase 2 để hệ thống truy xuất đúng tài liệu khi câu hỏi mới phụ thuộc vào lịch sử hội thoại.

Ví dụ:

- Lượt 1: “So sánh hai mô hình YOLO và Faster R-CNN.”
- Lượt 2: “Còn cái thứ hai thì sao?”

Nếu đưa nguyên câu hỏi ở lượt 2 vào retriever, hệ thống không biết “cái thứ hai” là gì. Ta cần viết lại câu hỏi thành câu độc lập, chẳng hạn: “Faster R-CNN có ưu điểm và nhược điểm gì?”

## 2. Luồng dữ liệu

1. Nhận `input` và `history`.
2. Nếu chưa có lịch sử, dùng trực tiếp `input` làm truy vấn — **không gọi LLM để viết lại**.
3. Nếu đã có lịch sử, gọi một prompt riêng để tạo câu hỏi độc lập.
4. Dùng câu hỏi độc lập để truy xuất tài liệu.
5. Tạo câu trả lời dựa trên `input` gốc, `history` và `context` lấy được.

Điểm quan trọng: câu hỏi đã viết lại chỉ phục vụ retrieval. Câu trả lời vẫn cần nhận câu hỏi gốc để giữ đúng ý định hiện tại của người dùng.

## 3. Cài đặt

Ví dụ dưới đây dùng LCEL và `RunnableBranch`. Giả sử Phase 2 đã có `retriever` — một retriever tương thích với LangChain — và `model` là chat model đã khởi tạo.

```bash
pip install -U langchain-core langchain
```

## 4. Code tích hợp

Tạo file `conversational_rag.py`, hoặc chuyển phần chain vào module RAG của Phase 2:

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.runnables.history import RunnableWithMessageHistory


# Được cung cấp bởi Phase 2:
# model = ...
# retriever = ...


# Bước 1: Viết lại câu hỏi thành câu độc lập.
contextualize_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        \"\"\"Dựa trên lịch sử hội thoại và câu hỏi mới, hãy viết lại câu hỏi
thành một câu độc lập để truy xuất tài liệu.

- Giữ nguyên ý định của câu hỏi mới.
- Chỉ bổ sung thông tin từ lịch sử khi cần để làm rõ tham chiếu.
- Không trả lời câu hỏi.
- Nếu câu hỏi đã độc lập, giữ nguyên ý nghĩa của nó.
- Chỉ xuất ra câu hỏi đã viết lại, không giải thích.\"\"\",
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

contextualize_chain = contextualize_prompt | model | StrOutputParser()


# Bước 2: Có history thì rewrite; không có history thì dùng input trực tiếp.
# Nhánh đầu tiên không gọi model.
contextual_question = RunnableBranch(
    (
        lambda x: not x.get("history"),
        RunnableLambda(lambda x: x["input"]),
    ),
    contextualize_chain,
)


# Bước 3: Retrieval dùng câu hỏi đã contextualize.
def retrieve_documents(x):
    query = x["retrieval_query"]
    return retriever.invoke(query)


# Bước 4: Answer dùng câu hỏi gốc + history + retrieved context.
answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        \"\"\"Bạn là trợ lý trả lời dựa trên tài liệu được truy xuất.
Ưu tiên thông tin trong CONTEXT; nếu không đủ bằng chứng, hãy nói rõ
là tài liệu hiện có chưa đủ thông tin. Không tự bịa nguồn.

CONTEXT:
{context}\"\"\",
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])


rag_core = (
    RunnablePassthrough.assign(
        retrieval_query=contextual_question
    )
    .assign(
        docs=retrieve_documents
    )
    .assign(
        context=lambda x: "\\n\\n".join(
            doc.page_content for doc in x["docs"]
        )
    )
    | answer_prompt
    | model
)


# history_factory_config có thể cấu hình tương tự Lesson 3.2.
# get_session_history phải trả về BaseChatMessageHistory.
conversational_rag = RunnableWithMessageHistory(
    rag_core,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)
```

### Cách gọi

```python
response = conversational_rag.invoke(
    {"input": "Còn cái thứ hai thì sao?"},
    config={
        "configurable": {
            "user_id": "user_001",
            "session_id": "session_001",
        }
    },
)

print(response.content)
```

**Lưu ý khi tích hợp:** thay `model`, `retriever` và `get_session_history` bằng các thành phần thật của dự án. Nếu Phase 2 đang dùng tên input hoặc schema khác, hãy điều chỉnh cho khớp. `retriever.invoke(query)` được giả định trả về danh sách `Document` có `page_content`.

## 5. Vì sao lượt đầu không tốn thêm một lần gọi LLM?

`RunnableBranch` kiểm tra `history` trước:

- `history` rỗng: nhánh `RunnableLambda` trả về câu hỏi gốc; không chạy `contextualize_chain`.
- `history` có dữ liệu: chạy `contextualize_chain`, tốn thêm một lần gọi LLM.

Lượt đầu vẫn cần các bước LLM vốn có trong RAG để sinh câu trả lời; chỉ **không phát sinh thêm lượt gọi LLM cho việc viết lại câu hỏi**.

## 6. Chi phí và lựa chọn thiết kế

- Contextualization làm tăng latency và token cost ở các lượt có lịch sử.
- Có thể dùng model nhỏ/rẻ hơn cho bước viết lại, còn model chính dùng để trả lời.
- Không nên đưa toàn bộ lịch sử quá dài vào prompt; dùng giới hạn lịch sử hoặc chiến lược rút gọn phù hợp.
- Agentic RAG là hướng khác: model quyết định khi nào và cách nào gọi retriever như một tool. Nội dung đó thuộc Phase 4.

## 7. Langfuse: theo dõi bước rewrite

Nên quan sát riêng span/trace của `contextualize_chain` và trace của toàn bộ RAG. Với mỗi lượt, cần kiểm tra:

- Câu hỏi gốc (`input`).
- Câu hỏi viết lại (`retrieval_query`).
- Thời gian chạy (latency) và token/cost của bước rewrite nếu provider cung cấp.
- Tài liệu được retrieve và câu trả lời cuối.

Nếu muốn hiển thị rõ câu hỏi gốc cạnh câu hỏi viết lại trong Langfuse, hãy log chúng dưới dạng input/output hoặc metadata của span. Không log dữ liệu nhạy cảm nếu không cần thiết.

## 8. Bài tập

Tích hợp contextual question rewriting vào RAG Phase 2 hiện tại.

### Checklist nghiệm thu

- [ ] Câu hỏi tiếp nối như “còn cái thứ hai thì sao?” được viết lại đủ ngữ cảnh để retrieval đúng.
- [ ] Lượt đầu không gọi LLM thêm cho bước rewrite.
- [ ] Câu trả lời cuối vẫn nhận câu hỏi gốc, lịch sử và context truy xuất.
- [ ] Chạy lại experiment của Phase 2 và so sánh chất lượng trước/sau; không để chất lượng giảm.
- [ ] Kiểm tra trace Langfuse để phân biệt câu hỏi gốc và câu hỏi viết lại.
- [ ] Kiểm tra trường hợp rewrite sai hoặc retrieval không tìm thấy tài liệu.

## 9. Gợi ý kiểm thử

Dùng cùng một bộ câu hỏi và tài liệu cho cả baseline Phase 2 lẫn phiên bản mới:

1. **Câu hỏi độc lập:** xác nhận kết quả tương đương baseline.
2. **Câu hỏi tiếp nối:** hỏi về một chủ đề, sau đó dùng “cái thứ hai”, “nó”, “so sánh với cái đó”.
3. **Lịch sử không liên quan:** xác nhận hệ thống không tự thêm thông tin không được hỏi.
4. **Không đủ bằng chứng:** xác nhận câu trả lời nói rõ giới hạn của context.
5. **Lượt đầu:** kiểm tra không có span/call contextualization.

Đừng chỉ đánh giá câu trả lời cuối; hãy kiểm tra cả truy vấn đã viết lại và danh sách tài liệu được retrieve.
