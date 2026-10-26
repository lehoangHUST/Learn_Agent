import os
os.environ.pop("SSL_CERT_FILE", None)
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

print("Đang tải Local Embedding Model... (Lần đầu sẽ mất thời gian để tải model từ Hugging Face)")
local_embeddings = HuggingFaceEmbeddings(
    model_name="keepitreal/vietnamese-sbert",
    model_kwargs={'device': 'cpu'}, 
    encode_kwargs={'normalize_embeddings': True} 
)

DB_DIR = "./data/local_chroma_db"
os.makedirs(DB_DIR, exist_ok=True)

vectorstore = Chroma(
    collection_name="local_rag_docs",
    embedding_function=local_embeddings,
    persist_directory=DB_DIR
)

if __name__ == "__main__":
    
    print("\n--- BƯỚC 1: NẠP DỮ LIỆU VÀO DATABASE ---")
    sample_docs = [
        Document(page_content="Công ty hỗ trợ ăn trưa 50,000 VND/ngày cho nhân viên chính thức.", metadata={"source": "HR_Policy", "page": 1}),
        Document(page_content="Chế độ nghỉ phép năm là 12 ngày có lương.", metadata={"source": "HR_Policy", "page": 2}),
        Document(page_content="Thưởng Tết nguyên đán tương đương 1 tháng lương thứ 13.", metadata={"source": "HR_Policy", "page": 3})
    ]
    
    # Hàm add_documents sẽ tự động gọi local_embeddings để chuyển text thành vector và lưu xuống ổ cứng
    # vectorstore.add_documents(sample_docs)
    # print(f"Đã lưu thành công {len(sample_docs)} tài liệu vào DB.")

    # Bước 2: Thử nghiệm biến câu hỏi thành Vector
    print("\n--- BƯỚC 2: KIỂM TRA EMBEDDING CÂU HỎI ---")
    query = "Trợ cấp ăn trưa công ty bao nhiêu?"
    vector_result = local_embeddings.embed_query(query)
    
    print(f"Câu query được chuyển thành vector có {len(vector_result)} chiều.")
    print(f"5 giá trị đầu tiên của vector: {vector_result[:5]}")

    # Bước 3: Tìm kiếm tài liệu liên quan nhất
    print("\n--- BƯỚC 3: TÌM KIẾM TƯƠNG ĐỒNG (RETRIEVAL) ---")
    # k=1 nghĩa là chỉ lấy 1 tài liệu có khoảng cách gần nhất với câu hỏi
    results = vectorstore.similarity_search_with_score(query, k=1)
    
    for doc, score in results:
        print(f"Câu hỏi của user: '{query}'")
        print(f"Tài liệu tìm thấy: '{doc.page_content}'")
        print(f"Score (Khoảng cách L2): {score:.4f} -> Giá trị càng sát 0 càng giống nhau.")