"""
Lesson2.1 hiện tại đang chuẩn hóa tài liệu từ đuôi miền khác nhau -> Quy về dạng Document trong Langchain
"""
import os
import datetime
import hashlib
import unicodedata
from typing import Iterator
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader
)

def generate_checksum(content: str) -> str:
    """
    Create hash MD5
    """
    return hashlib.md5(content.encode("utf-8")).hexdigest()

def normalize_vietnamese_text(text: str) -> str:
    if not text:
        return ""
    
    normalize_text = unicodedata.normalize("NFC", text)
    
    clean_text = " ".join(normalize_text.split())
    return clean_text

def process_document(raw_doc: Document) -> Document:
    
    clean_document = normalize_vietnamese_text(raw_doc.page_content)
    original_metadata = raw_doc.metadata
    
    source = original_metadata.get("source", "unknown")
    page = original_metadata.get("page", 0) # PyPDFLoader tự thêm trường page
    
    title = os.path.basename(source) if source != "unknown" else "Untitled"
    doc_type = os.path.splitext(source)[1].replace(".", "").lower() if source != "unknown" else "unknown"

    enriched_metadata = {
        "source": source,
        "page": page,
        "title": title,
        "doc_type": doc_type,
        "updated_at": datetime.now().isoformat(),
        "checksum": generate_checksum(clean_document),
        "is_trusted": False
    }
    
    return Document(
        page_content=clean_document,
        metadata=enriched_metadata,
        id=enriched_metadata["checksum"]
    )

def safe_ingest_directory(directory_path: str, glob_pattern: str = "**/*.pdf") -> Iterator[Document]:
    
    # use PyPDFLoader for file PDF
    loader = DirectoryLoader(
        directory_path,
        glob=glob_pattern,
        loader_cls=PyPDFLoader,
        show_progress=True,
        silent_errors=False
    )
    
    # use lazy_load
    try:
        doc_iter = loader.lazy_load()
        for raw_doc in doc_iter:
            try:
                processed_doc = process_document(raw_doc)
                
                if len(processed_doc.page_content) > 10:
                    yield processed_doc
                else:
                    print(f"Bỏ qua tài liệu quá ngắn hoặc rỗng từ nguồn: {raw_doc.metadata.get('source')}")
            except Exception as e:
                print(f"Lỗi khi xử lý tài liệu {raw_doc.metadata.get('source')}: {str(e)}")
                continue
            
            break
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi đọc thư mục {directory_path}: {str(e)}")
        
def ingest_mixed_directory(directory_path: str) -> Iterator[Document]:
    
    clean_documents = []
    loader_mapping = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader
    }
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            
            if ext not in loader_mapping:
                continue
            
            doc_type = ext.replace(".", "")
            loader_class = loader_mapping[ext]
            
            try:
                if loader_class == TextLoader:
                    loader = loader_class(file_path, autodetect_encoding=True)
                else:
                    loader = loader_class(file_path)
                    
                for raw_doc in loader.lazy_load():
                    try:
                        processed_doc = process_document(
                            raw_doc,
                            file_path,
                            doc_type
                        )
                        
                        if len(processed_doc.page_content) > 5:
                            yield processed_doc
                        
                    except Exception as err:
                        print(f"[Error] Normalize text: {err}")
                    
            except Exception as err:
                print(f"[Error] Read file error: {err}")
                continue
        
if __name__ == "__main__":
    directory_path = "C:/Users/hoang/OneDrive/Documents/Code/Learn_Agent/Lesson2/example-docs"
    safe_ingest_directory(directory_path)