"""
verify_judge.py
Kiểm tra độ chính xác của LLM-as-a-judge trên 5 mẫu nhãn tay trước khi chạy batch.
"""
import os
os.environ.pop("SSL_CERT_FILE", None)
from eval_metrics import judge_faithfulness, judge_answer_relevance

BENCHMARK_SAMPLES = [
    {
        "name": "Case 1: Hoàn hảo (Faithful & Relevant)",
        "query": "Giờ làm việc của công ty?",
        "context": "Thời gian làm việc từ thứ 2 đến thứ 6, 8h30 đến 17h30.",
        "answer": "Giờ làm việc là từ thứ 2 đến thứ 6, bắt đầu lúc 8h30 và kết thúc lúc 17h30.",
        "expected_answer": "Thứ 2 đến thứ 6, 8h30 - 17h30.",
        "gt_faithfulness": 1.0,
        "gt_relevance": 1.0
    },
    {
        "name": "Case 2: Ảo giác (Hallucination - Bịa thêm thông tin)",
        "query": "Quy định WFH thế nào?",
        "context": "Nhân viên gửi yêu cầu WFH cho Trưởng bộ phận duyệt.",
        "answer": "Nhân viên gửi yêu cầu cho Trưởng bộ phận duyệt và được làm việc ở bãi biển tối đa 1 tháng.",
        "expected_answer": "Trưởng bộ phận trực tiếp phê duyệt.",
        "gt_faithfulness": 0.0,
        "gt_relevance": 0.5
    },
    {
        "name": "Case 3: Lạc đề (Irrelevant)",
        "query": "Hạn nộp chi phí công tác là khi nào?",
        "context": "Hạn nộp là trong vòng 5 ngày làm việc sau chuyến đi.",
        "answer": "Công ty hỗ trợ 150.000 VNĐ tiền ăn trưa cho nhân viên đi công tác.",
        "expected_answer": "Trong vòng 5 ngày làm việc.",
        "gt_faithfulness": 1.0, # Trung thực với tài liệu nhưng lạc đề
        "gt_relevance": 0.0
    },
    {
        "name": "Case 4: Từ chối chuẩn (Correct Refusal)",
        "query": "Giá vé máy bay đi Tokyo là bao nhiêu?",
        "context": "",
        "answer": "Tôi không tìm thấy thông tin này trong tài liệu nội bộ.",
        "expected_answer": "Không có thông tin.",
        "gt_faithfulness": 1.0,
        "gt_relevance": 1.0
    },
    {
        "name": "Case 5: Trả lời sai sự thật (Factually Wrong)",
        "query": "Phép năm có bao nhiêu ngày?",
        "context": "Mỗi nhân viên chính thức có 12 ngày phép năm.",
        "answer": "Nhân viên được nghỉ 25 ngày phép năm có lương.",
        "expected_answer": "12 ngày phép năm.",
        "gt_faithfulness": 0.0,
        "gt_relevance": 0.0
    }
]

def run_verification():
    print("=" * 60)
    print("BẮT ĐẦU KIỂM ĐỊNH LLM-AS-A-JUDGE TRÊN 5 MẪU GÁN NHÃN TAY")
    print("=" * 60)
    
    passed_cases = 0
    for idx, sample in enumerate(BENCHMARK_SAMPLES, start=1):
        faith_pred = judge_faithfulness(sample["query"], sample["context"], sample["answer"])
        rel_pred = judge_answer_relevance(sample["query"], sample["answer"], sample["expected_answer"])
        
        faith_ok = (faith_pred == sample["gt_faithfulness"])
        rel_ok = (rel_pred == sample["gt_relevance"])
        
        status = "PASSED" if (faith_ok and rel_ok) else "FAILED"
        if status == "PASSED":
            passed_cases += 1
            
        print(f"[{status}] {sample['name']}")
        print(f"   - Faithfulness: Dự đoán={faith_pred} | Nhãn tay={sample['gt_faithfulness']}")
        print(f"   - Relevance:    Dự đoán={rel_pred} | Nhãn tay={sample['gt_relevance']}")
        print("-" * 50)

    print(f"Kết quả: {passed_cases}/5 mẫu khớp hoàn toàn với Ground Truth nhãn tay.")
    if passed_cases >= 4:
        print(">> Judge đạt tiêu chuẩn kiểm định! Sẵn sàng chạy batch.")
    else:
        print(">> CẢNH BÁO: Cần xem lại Prompt Rubric trước khi chạy batch.")

if __name__ == "__main__":
    run_verification()