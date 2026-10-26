import hashlib
import tiktoken
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Khởi tạo bộ đếm token chuẩn xác của OpenAI
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def generate_stable_chunk_id(source: str, page: int, content: str) -> str:
    """
    PASS ĐIỀU KIỆN 1: Chunk ID ổn định.
    Hash MD5 dựa trên Nội dung + Tên file + Trang.
    Nếu chạy lại 1000 lần, ID này vẫn không đổi.
    """
    unique_string = f"{source}_{page}_{content}"
    return hashlib.md5(unique_string.encode('utf-8')).hexdigest()

def evaluate_chunking_config(config_name: str, chunks: list[Document], original_tokens: int):
    """Hàm thống kê số liệu để đưa ra quyết định."""
    if not chunks:
        return
        
    token_lengths = [count_tokens(doc.page_content) for doc in chunks]
    total_chunk_tokens = sum(token_lengths)
    
    # Tính tỉ lệ trùng lặp (Overlap Ratio)
    overlap_ratio = ((total_chunk_tokens / original_tokens) - 1) * 100 if original_tokens else 0

    print(f"\n{'='*40}")
    print(f"📊 BÁO CÁO CẤU HÌNH: {config_name}")
    print(f"{'='*40}")
    print(f"1. Tổng số chunks: {len(chunks)}")
    print(f"2. Phân bố Token:")
    print(f"   - Nhỏ nhất (Min): {min(token_lengths)} tokens")
    print(f"   - Lớn nhất (Max): {max(token_lengths)} tokens")
    print(f"   - Trung bình (Avg): {np.mean(token_lengths):.1f} tokens")
    print(f"3. Tỉ lệ phình dữ liệu (Overlap): {overlap_ratio:.1f}%")
    
    # PASS ĐIỀU KIỆN 3: Mọi chunk đều còn metadata nguồn
    sample_chunk = chunks[0]
    print(f"\n🔍 Kiểm chứng Metadata (Chunk 0):")
    print(f"   - Nguồn: {sample_chunk.metadata.get('source')}")
    print(f"   - Trang: {sample_chunk.metadata.get('page')}")
    print(f"   - Chunk ID: {sample_chunk.id}")

# ================= KỊCH BẢN THỬ NGHIỆM =================

if __name__ == "__main__":
    # Giả lập 1 Document đã qua bước Load & Normalize (Lesson 2.1)
    # Nội dung đủ dài để splitter phải thực hiện cắt nhiều lần
    base_text = (
        "Trí tuệ nhân tạo (AI) đang thay đổi cách chúng ta làm việc. " * 30 + "\n\n" +
        "Tại Việt Nam, việc áp dụng RAG đòi hỏi kỹ thuật xử lý dữ liệu chuẩn xác. " * 30 + "\n\n" +
        "Chunking quyết định trực tiếp đến chất lượng tìm kiếm của Vector Database. " * 30
    )
    
    original_doc = Document(
        page_content=base_text, 
        metadata={"source": "RAG_Guidelines.pdf", "page": 1}
    )
    
    original_token_count = count_tokens(base_text)
    print(f"Tài liệu gốc: {original_token_count} tokens")

    # --- Cấu hình 1: Theo ký tự (Dễ gây lệch token với tiếng Việt) ---
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        length_function=len
    )

    # --- Cấu hình 2: Theo Token - Nhỏ (Chính xác cho truy vấn cụ thể) ---
    token_splitter_small = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=150,
        chunk_overlap=30
    )

    # --- Cấu hình 3: Theo Token - Lớn (Tốt cho LLM cần ngữ cảnh rộng) ---
    token_splitter_large = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=300,
        chunk_overlap=50
    )

    configs = [
        ("Cắt theo Ký tự (Size 600c, Overlap 100c)", char_splitter),
        ("Cắt theo Token NHỎ (Size 150t, Overlap 30t)", token_splitter_small),
        ("Cắt theo Token LỚN (Size 300t, Overlap 50t)", token_splitter_large)
    ]

    # Thực thi và gán ID
    for name, splitter in configs:
        # split_documents tự động giữ nguyên metadata của original_doc
        chunks = splitter.split_documents([original_doc])
        
        # Gán ID ổn định cho từng chunk
        for doc in chunks:
            doc.id = generate_stable_chunk_id(
                source=doc.metadata["source"], 
                page=doc.metadata["page"], 
                content=doc.page_content
            )
            
        evaluate_chunking_config(name, chunks, original_token_count)