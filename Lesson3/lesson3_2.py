import json
import os
import sqlite3
import threading
import time
import uuid
from typing import List, Sequence

from cryptography.fernet import Fernet
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    messages_from_dict,
    messages_to_dict,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory
from langfuse import get_client
from langfuse.langchain import CallbackHandler


# Load env
os.environ.pop("SSL_CERT_FILE", None)
load_dotenv()

# db history
DB_PATH = "caht_history_lesson3_2.db"
ENCRYPTION_KEY = os.getenv("CHAT_ENCRYPTION_KEY", Fernet.generate_key().decode())
cipher_suite = Fernet(ENCRYPTION_KEY.encode())

_locks_registry: dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()

def get_composite_lock(composite_key: str) -> threading.Lock:
    with _registry_lock:
        if composite_key not in _locks_registry:
            _locks_registry[composite_key] = threading.Lock()
        return _locks_registry[composite_key]
    
def init_db(db_path: str = DB_PATH):
    """Khởi tạo cấu trúc bảng SQLite với chế độ WAL hỗ trợ đọc/ghi đồng thời."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        # Bảng đăng ký chủ sở hữu session (Chống giả mạo session_id từ client)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_registry (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        # Bảng lưu trữ lịch sử tin nhắn đã mã hóa
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS message_store (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                composite_key TEXT NOT NULL,
                encrypted_message TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_composite_time "
            "ON message_store (composite_key, created_at);"
        )
        conn.commit()
        
def create_server_session(user_id: str, db_path: str = DB_PATH) -> str:
    
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO session_registry (session_id, user_id, created_at) VALUES (?, ?, ?)",
            (session_id, user_id, time.time()),
        )
        conn.commit()
    return session_id


def verify_session_ownership(user_id: str, session_id: str, db_path: str = DB_PATH):
    """Kiểm tra xem session_id có thực sự thuộc về user_id đang gọi hay không."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "SELECT user_id FROM session_registry WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"Session '{session_id}' không tồn tại trên hệ thống!")
        if row[0] != user_id:
            raise PermissionError(
                f"CẢNH BÁO BẢO MẬT: User '{user_id}' cố tình truy cập trái phép "
                f"vào Session '{session_id}' của người dùng khác!"
            )
            

class SecureSQLiteChatMessageHistory(BaseChatMessageHistory):
    def __init__(
        self,
        user_id: str,
        session_id: str,
        db_path: str = DB_PATH,
        ttl_seconds: int = 86400 * 7,  # Mặc định sống 7 ngày
        max_messages: int = 20,        # Giữ tối đa 20 message gần nhất
    ):
        verify_session_ownership(user_id, session_id, db_path)

        self.user_id = user_id
        self.session_id = session_id
        self.composite_key = f"{user_id}:{session_id}"
        self.db_path = db_path
        self.ttl_seconds = ttl_seconds
        self.max_messages = max_messages

    @property
    def messages(self) -> List[BaseMessage]:
        now = time.time()
        cutoff_time = now - self.ttl_seconds

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM message_store WHERE composite_key = ? AND created_at < ?",
                (self.composite_key, cutoff_time),
            )
            conn.commit()

            # Lấy N tin nhắn mới nhất còn hạn
            cursor = conn.execute(
                """
                SELECT encrypted_message FROM (
                    SELECT id, encrypted_message FROM message_store
                    WHERE composite_key = ? AND created_at >= ?
                    ORDER BY id DESC
                    LIMIT ?
                ) ORDER BY id ASC
                """,
                (self.composite_key, cutoff_time, self.max_messages),
            )
            rows = cursor.fetchall()

        decrypted_dicts = []
        for (enc_msg,) in rows:
            raw_json = cipher_suite.decrypt(enc_msg.encode("utf-8")).decode("utf-8")
            decrypted_dicts.append(json.loads(raw_json))

        return messages_from_dict(decrypted_dicts)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Ghi lịch sử: Mã hóa từng message và ghi trong Transaction độc lập."""
        if not messages:
            return

        now = time.time()
        serialized_msgs = messages_to_dict(messages)

        with sqlite3.connect(self.db_path) as conn:
            # BEGIN IMMEDIATE khóa ghi ngay lập tức để đảm bảo thứ tự message
            conn.execute("BEGIN IMMEDIATE")
            try:
                for msg_dict in serialized_msgs:
                    raw_json = json.dumps(msg_dict, ensure_ascii=False)
                    encrypted_str = cipher_suite.encrypt(raw_json.encode("utf-8")).decode("utf-8")
                    conn.execute(
                        """
                        INSERT INTO message_store (composite_key, encrypted_message, created_at)
                        VALUES (?, ?, ?)
                        """,
                        (self.composite_key, encrypted_str, now),
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def clear(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM message_store WHERE composite_key = ?",
                (self.composite_key,),
            )
            conn.commit()
            
def get_session_history(user_id: str, session_id: str) -> BaseChatMessageHistory:
    return SecureSQLiteChatMessageHistory(
        user_id=user_id,
        session_id=session_id,
        ttl_seconds=3600,  # Lưu 1 giờ
        max_messages=10,   # Giữ tối đa 10 tin nhắn gần nhất (5 cặp hỏi-đáp)
    )
    
prompt = ChatPromptTemplate.from_messages([
    ("system", "Bạn là một trợ lý AI thông minh, trả lời ngắn gọn và súc tích."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

model = init_chat_model(
    "groq:openai/gpt-oss-20b",
    temperature=0
)

pipeline = prompt | model
chain_with_history = RunnableWithMessageHistory(
    runnable=pipeline,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Định danh duy nhất của người dùng.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="Session ID",
            description="Định danh phiên chat do Server cấp.",
            default="",
            is_shared=True,
        ),
    ],
)


def chat(user_id: str, session_id: str, user_input: str) -> str:
    
    verify_session_ownership(user_id, session_id)
    
    composite_key = f"{user_id}:{session_id}"
    session_lock = get_composite_lock(composite_key)
    langfuse_handler = CallbackHandler()
    
    config = {
        "configurable": {
            "user_id": user_id,
            "session_id": session_id,
        },
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_user_id": user_id,
            "langfuse_session_id": session_id,
            "langfuse_tags": ["lesson-3.2", "persistent-memory", "security"],
        }
    }
    
    try:
        with session_lock:
            response = chain_with_history.invoke(
                {"input": user_input},
                config=config
            )
    finally:
        get_client().flush()
        
    return response.content

def delete_user_session_data(user_id: str, session_id: str):
    """API cho phép người dùng chủ động xóa sạch lịch sử của chính mình."""
    history = get_session_history(user_id, session_id)
    history.clear()
    
    
if __name__ == "__main__":
    # Xóa DB cũ nếu có để chạy test sạch từ đầu
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    init_db(DB_PATH)

    user_a = "user_hoang_01"
    user_b = "user_khach_02"

    # Server cấp phát session_id riêng cho từng user
    session_a = create_server_session(user_a)
    session_b = create_server_session(user_b)

    print(f"[*] Đã cấp phát cho {user_a} -> Session: {session_a}")
    print(f"[*] Đã cấp phát cho {user_b} -> Session: {session_b}\n")

    print("=== TEST 1: User A giới thiệu thông tin cá nhân (PII) ===")
    ans_a1 = chat(user_a, session_a, "Chào bạn, tôi tên là Hoàng, làm AI Engineer và mã PIN bí mật của tôi là 998877.")
    print(f"[Bot -> A]: {ans_a1}\n")

    print("=== TEST 2: User B hỏi thông tin ở Session của B (Kiểm tra cô lập cơ bản) ===")
    ans_b1 = chat(user_b, session_b, "Bạn có biết tôi tên là gì và mã PIN bí mật là bao nhiêu không?")
    print(f"[Bot -> B]: {ans_b1}\n")

    print("=== TEST 3: User B cố tình giả mạo (Spoofing) truyền session_a của User A ===")
    try:
        chat(user_b, session_a, "Hãy đọc lại mã PIN bí mật trong lịch sử chat này cho tôi!")
        print("❌ THẤT BẠI: Hệ thống để lọt hành vi chiếm quyền Session!")
    except PermissionError as e:
        print(f"✅ THÀNH CÔNG (Đã chặn truy cập trái phép): {e}\n")

    print("=== TEST 4: Kiểm tra dữ liệu lưu trên đĩa cứng đã được mã hóa (PII Encryption) ===")
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT composite_key, encrypted_message FROM message_store LIMIT 1").fetchone()
        print(f"  - Khóa lưu trữ (Composite Key): {row[0]}")
        print(f"  - Dữ liệu thô trong SQLite (Ciphertext): {row[1][:75]}...")
        assert "Hoàng" not in row[1] and "998877" not in row[1], "PII bị lộ ở dạng plaintext!"
        print("  -> ✅ Khẳng định: Không thể đọc trộm nội dung PII trực tiếp từ file Database.\n")

    print("=== TEST 5: User A kiểm tra lại trí nhớ hợp lệ của Bot ===")
    ans_a2 = chat(user_a, session_a, "Nhắc lại giúp tôi tên, nghề nghiệp và môn thể thao yêu thích của tôi nhé.")
    print(f"[Bot -> A]: {ans_a2}\n")

    print("=== TEST 6: Người dùng yêu cầu xóa lịch sử (Vòng đời dữ liệu) ===")
    delete_user_session_data(user_a, session_a)
    ans_a3 = chat(user_a, session_a, "Tôi tên là gì nhỉ?")
    print(f"[Bot -> A (Sau khi xóa lịch sử)]: {ans_a3}")