import os

import threading
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langfuse import get_client
from langfuse.langchain import CallbackHandler


# load env
os.environ.pop("SSL_CERT_FILE", None)
load_dotenv()

# init store in RAM: {"session_id": InMemoryChatMessageHistory}
store: dict[str, InMemoryChatMessageHistory] = {}
_session_locks = {}
_store_lock = threading.Lock()

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    with _store_lock:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]
    

def get_session_lock(session_id: str):
    with _store_lock:
        if session_id not in _session_locks:
            _session_locks[session_id] = threading.Lock()
            
        return _session_locks[session_id]
    
prompt = ChatPromptTemplate.from_messages([
    ("system", "Bạn là một trợ lý AI thông minh, trả lời ngắn gọn và súc tích."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

model = init_chat_model(
    "groq:openai/gpt-oss-20b",
    temperature=0
)

# runnable with prompt -> model
pipeline = prompt | model

chain_with_history = RunnableWithMessageHistory(
    runnable=pipeline,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

def chat(session_id: str, user_input: str) -> str:
    """
    Process chat by session_id
    """
    langfuse_handler = CallbackHandler()
    
    config = {
        "configurable": {"session_id": session_id},
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_session_id": session_id,
            "langfuse_tags": ["lesson-3.1", "lcel-memory"]
        }
    }

    session_lock = get_session_lock(session_id)

    try:
        with session_lock:
            response = chain_with_history.invoke(
                {"input": user_input},
                config=config,
            )
    finally:
        get_client().flush()

    return response.content

if __name__ == "__main__":
    
    session_a = "name_a"
    session_b = "name_b"
    
    print("=== LƯỢT 1: User A giới thiệu thông tin ===")
    ans_a1 = chat(session_a, "Chào bạn, tôi tên là Hoàng và tôi đang làm AI Engineer.")
    print(f"[Bot -> A]: {ans_a1}\n")
    
    print("=== LƯỢT 2: User B hỏi thông tin của User A (Kiểm tra cô lập Session) ===")
    ans_b1 = chat(session_b, "Chào bạn, bạn có biết tôi tên là gì và làm nghề gì không?")
    print(f"[Bot -> B]: {ans_b1}\n")
    
    print("=== LƯỢT 3: User A kiểm tra trí nhớ của Bot ===")
    ans_a2 = chat(session_a, "Tôi tên là gì và làm nghề gì nhỉ?")
    print(f"[Bot -> A]: {ans_a2}\n")

    # In kiểm tra trạng thái lưu trữ thực tế bên trong biến `store`
    print("=== KIỂM TRA DỮ LIỆU TRONG MEMORY STORE ===")
    for sid, history in store.items():
        print(f"\n--- Lịch sử của {sid} ({len(history.messages)} messages) ---")
        for msg in history.messages:
            role = "Human" if msg.type == "human" else "AI"
            print(f"  [{role}]: {msg.content}")