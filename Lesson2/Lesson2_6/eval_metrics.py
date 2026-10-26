"""
eval_metrics.py
Chứa các hàm tính toán retrieval metrics (code) và generation metrics (LLM-as-a-judge).
"""
import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

def compute_hit_rate(retrieved_chunk_ids: List[str], expected_chunk_ids: List[str]) -> float:
    """Hit Rate@k: Bằng 1.0 nếu có ít nhất 1 chunk kỳ vọng nằm trong top k; ngược lại 0.0.
    Nếu câu hỏi unanswerable (expected trống), coi như 1.0 nếu hệ thống trả về kết quả rỗng,
    hoặc 1.0 (không áp dụng retrieval penalty). Ở đây unanswerable đánh giá riêng qua refusal.
    """
    if not expected_chunk_ids:
        return 1.0  # Không có chunk kỳ vọng (OOD)
    hits = set(retrieved_chunk_ids).intersection(set(expected_chunk_ids))
    return 1.0 if len(hits) > 0 else 0.0

def compute_mrr(retrieval_chunk_ids: List[str], expected_chunk_ids: List[str]) -> float:
    """MRR (Mean Reciprocal Rank): 1 / rank"""
    if not expected_chunk_ids:
        return 1.0
    
    for rank, chunk_id in enumerate(retrieval_chunk_ids, start=1):
        if chunk_id in expected_chunk_ids:
            return 1.0 / rank
    return 0.0
    

# GENERATION METRIC (LLM-as-a-Judge with Structured Output)

class FaithfulnessAssesment(BaseModel):
    reasoning: str = Field(description="Phân tích từng câu khẳng định xem có được context hỗ trợ không")
    score: float = Field(description="1.0 nếu hoàn toàn trung thực với context, 0.0 nếu có ảo giác (hallucination)")
    

class AnswerRelevanceAssessment(BaseModel):
    reasoning: str = Field(description="Giải thích câu trả lời có trực diện trả lời query không")
    score: float = Field(description="1.0 nếu trả lời đúng trọng tâm câu hỏi, 0.0 nếu lệch đề hoặc trả lời sai")
    

model = init_chat_model(
    "groq:openai/gpt-oss-20b",
    temperature=0
)
structured_judge = model.with_structured_output(FaithfulnessAssesment) # struct judge
answer_relevance_judge = model.with_structured_output(AnswerRelevanceAssessment) # answer relevance

   
def judge_faithfulness(query: str, context: str, answer: str) -> float:
    """Đánh giá xem câu trả lời có bị hallucination so với context hay không."""
    if not context.strip():
        # Nếu không có context mà model tự chém -> 0.0, nếu model từ chối -> 1.0
        refusal_keywords = ["không tìm thấy", "không có thông tin", "chưa đủ dữ liệu"]
        return 1.0 if any(kw in answer.lower() for kw in refusal_keywords) else 0.0
    
    prompt = f"""Bạn là một chuyên gia đánh giá RAG khách quan.
        Nhiệm vụ: Hãy kiểm tra xem câu trả lời (Answer) có hoàn toàn dựa trên Ngữ cảnh (Context) đã cung cấp hay không.
        Nếu câu trả lời có chứa thông tin không thể suy ra từ Context, hãy cho điểm 0.0.
        Nếu mọi thông tin đều được hỗ trợ bởi Context, hãy cho điểm 1.0.

        [Context]:
        {context}

        [Answer]:
        {answer}
    """
    
    try:
        result: FaithfulnessAssesment = structured_judge.invoke(prompt)
        return result.score
        
    except Exception as err:
        print(f"Error Faithfulness: {err}")
        return 0.0
    
def judge_answer_relevance(query: str, answer: str, expected_answer: str) -> float:
    
    prompt = f"""Bạn là chuyên gia thẩm định câu trả lời.
        Nhiệm vụ: So sánh câu trả lời của hệ thống (System Answer) với Đáp án tham chiếu (Expected Answer) đối với Câu hỏi (Query).
        Rubric:
        - 1.0: Câu trả lời truyền tải đúng ý nghĩa trọng tâm của đáp án tham chiếu, ngắn gọn, súc tích.
        - 0.5: Trả lời đúng một phần nhưng thiếu ý quan trọng.
        - 0.0: Trả lời sai, lạc đề, hoặc ảo giác.

        [Query]: {query}
        [Expected Answer]: {expected_answer}
        [System Answer]: {answer}
    """
    try:
        result: FaithfulnessAssesment = answer_relevance_judge.invoke(prompt)
        return result.score
        
    except Exception as err:
        print(f"Error Faithfulness: {err}")
        return 0.0
    
def judge_refusal_accuracy(category: str, answer: str) -> Optional[float]:
    """Kiểm tra xem hệ thống có từ chối trả lời đúng lúc hay không (dành cho category unanswerable)."""
    if category != "unanswerable":
        return None
    refusal_markers = ["không tìm thấy", "không có thông tin", "không được đề cập", "tôi không biết"]
    is_refused = any(marker in answer.lower() for marker in refusal_markers)
    return 1.0 if is_refused else 0.0