import os
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.api import NotFoundError

os.environ.pop("SSL_CERT_FILE", None)
load_dotenv()

# Phải tạo client SAU load_dotenv(), nếu không sẽ không đọc được LANGFUSE_* keys
langfuse = Langfuse()

DATASET_NAME = "rag_eval_dataset_v1"

EVAL_ITEMS = [
    # --- CÂU DỄ (Factoid retrieval) ---
    {
        "query": "Chính sách nghỉ phép năm của công ty là bao nhiêu ngày?",
        "expected_answer": "Nhân viên chính thức được hưởng 12 ngày phép năm có hưởng lương.",
        "expected_chunk_ids": ["doc_hr_01"],
        "category": "easy"
    },
    {
        "query": "Thời gian làm việc tiêu chuẩn hàng tuần là gì?",
        "expected_answer": "Từ thứ 2 đến thứ 6, 8h30 đến 17h30 (nghỉ trưa 1 tiếng).",
        "expected_chunk_ids": ["doc_hr_02"],
        "category": "easy"
    },
    {
        "query": "Hạn nộp báo cáo chi phí công tác là khi nào?",
        "expected_answer": "Trong vòng 5 ngày làm việc kể từ khi kết thúc chuyến công tác.",
        "expected_chunk_ids": ["doc_finance_01"],
        "category": "easy"
    },
    {
        "query": "Mức hoàn trả tiền ăn trưa công tác tối đa là bao nhiêu?",
        "expected_answer": "Mức tối đa là 150.000 VNĐ/bữa.",
        "expected_chunk_ids": ["doc_finance_02"],
        "category": "easy"
    },
    {
        "query": "Quy định về mật khẩu hệ thống nội bộ là gì?",
        "expected_answer": "Mật khẩu tối thiểu 12 ký tự, gồm chữ hoa, thường, số và ký tự đặc biệt, đổi mỗi 90 ngày.",
        "expected_chunk_ids": ["doc_sec_01"],
        "category": "easy"
    },
    {
        "query": "Ai là người phê duyệt yêu cầu làm việc từ xa (WFH)?",
        "expected_answer": "Trưởng bộ phận trực tiếp và gửi thông báo cho HR.",
        "expected_chunk_ids": ["doc_hr_03"],
        "category": "easy"
    },
    {
        "query": "Thời gian thử việc tối đa cho vị trí kỹ sư phần mềm là bao lâu?",
        "expected_answer": "Thời gian thử việc tối đa là 60 ngày theo luật lao động.",
        "expected_chunk_ids": ["doc_hr_04"],
        "category": "easy"
    },

    # --- CÂU KHÓ (Cần ngữ cảnh rộng, tổng hợp thông tin) ---
    {
        "query": "Nếu nhân viên đi công tác vào ngày thứ Bảy thì được tính phụ cấp và nghỉ bù như thế nào?",
        "expected_answer": "Được hưởng 200% phụ cấp ngày làm việc ngoài giờ và được đăng ký 01 ngày nghỉ bù trong tháng kế tiếp.",
        "expected_chunk_ids": ["doc_hr_05", "doc_finance_03"],
        "category": "hard"
    },
    {
        "query": "Quy trình xử lý khi phát hiện laptop công ty bị nhiễm mã độc tống tiền (Ransomware)?",
        "expected_answer": "Lập tức ngắt kết nối mạng (WiFi và dây LAN), báo ngay cho SOC qua hotline 1900-xxxx, không khởi động lại máy.",
        "expected_chunk_ids": ["doc_sec_02", "doc_sec_03"],
        "category": "hard"
    },
    {
        "query": "Gói bảo hiểm sức khỏe nâng cao áp dụng cho những đối tượng nào và có quyền lợi gì đặc biệt?",
        "expected_answer": "Áp dụng cho nhân viên từ cấp Senior hoặc thâm niên trên 2 năm, bao gồm bảo lãnh nội trú và nha khoa hạn mức 30 triệu/năm.",
        "expected_chunk_ids": ["doc_hr_06"],
        "category": "hard"
    },
    {
        "query": "Chính sách đào tạo cấp chứng chỉ AWS/GCP yêu cầu cam kết làm việc bao lâu sau khi thi đỗ?",
        "expected_answer": "Nếu chi phí trên 10 triệu VNĐ, nhân viên cam kết làm việc tối thiểu 12 tháng, dưới 10 triệu cam kết 6 tháng.",
        "expected_chunk_ids": ["doc_hr_07"],
        "category": "hard"
    },
    {
        "query": "So sánh quy trình mua sắm thiết bị IT dưới 5 triệu và trên 5 triệu VNĐ.",
        "expected_answer": "Dưới 5 triệu chỉ cần Team Lead duyệt ticket; trên 5 triệu cần báo giá 3 bên và phê duyệt của IT Director.",
        "expected_chunk_ids": ["doc_it_01", "doc_finance_04"],
        "category": "hard"
    },
    {
        "query": "Quy định bảo mật source code khi làm việc tại quán cà phê hoặc không gian công cộng?",
        "expected_answer": "Bắt buộc dùng VPN công ty, bật tấm lọc chống nhìn trộm (privacy screen), không dùng WiFi công cộng mở không có mật khẩu.",
        "expected_chunk_ids": ["doc_sec_04"],
        "category": "hard"
    },
    {
        "query": "Nhân viên nữ sinh con được hưởng chế độ thai sản bổ sung nào từ quỹ nội bộ ngoài BHXH?",
        "expected_answer": "Được nhận khoản trợ cấp một lần 5.000.000 VNĐ từ công đoàn và quà tặng mừng em bé trị giá 1.000.000 VNĐ.",
        "expected_chunk_ids": ["doc_hr_08"],
        "category": "hard"
    },

    # --- CÂU KHÔNG CÓ TRONG TÀI LIỆU (Out-of-domain / Refusal test) ---
    {
        "query": "Công ty có hỗ trợ vé máy bay đi du lịch cá nhân sang Nhật Bản không?",
        "expected_answer": "Tôi không tìm thấy thông tin này trong tài liệu nội bộ.",
        "expected_chunk_ids": [],
        "category": "unanswerable"
    },
    {
        "query": "Thực đơn căn tin công ty vào thứ Tư tuần sau có món gì?",
        "expected_answer": "Tôi không tìm thấy thông tin này trong tài liệu nội bộ.",
        "expected_chunk_ids": [],
        "category": "unanswerable"
    },
    {
        "query": "Tỷ giá quy đổi Bitcoin sang VNĐ hôm nay là bao nhiêu?",
        "expected_answer": "Tôi không tìm thấy thông tin này trong tài liệu nội bộ.",
        "expected_chunk_ids": [],
        "category": "unanswerable"
    },
    {
        "query": "Chính sách thưởng cổ phiếu ESOP năm 2030 là như thế nào?",
        "expected_answer": "Tôi không tìm thấy thông tin này trong tài liệu nội bộ.",
        "expected_chunk_ids": [],
        "category": "unanswerable"
    },
    {
        "query": "Làm thế nào để xin giảm trừ thuế thu nhập cá nhân cho bố mẹ ở nước ngoài?",
        "expected_answer": "Tôi không tìm thấy thông tin này trong tài liệu nội bộ.",
        "expected_chunk_ids": [],
        "category": "unanswerable"
    },
    {
        "query": "Văn phòng chi nhánh tại New York mở cửa lúc mấy giờ?",
        "expected_answer": "Tôi không tìm thấy thông tin này trong tài liệu nội bộ.",
        "expected_chunk_ids": [],
        "category": "unanswerable"
    }
]

def main():
    
    print(f"Đang đồng bộ dataset '{DATASET_NAME}' lên Langfuse...")
    try:
        langfuse.get_dataset(DATASET_NAME)
    except NotFoundError:
        langfuse.create_dataset(name=DATASET_NAME, description="Dataset đánh giá RAG Lesson 2.6")

    for idx, item in enumerate(EVAL_ITEMS, start=1):
        # id cố định -> chạy lại sẽ cập nhật item cũ thay vì tạo bản trùng
        langfuse.create_dataset_item(
            id=f"{DATASET_NAME}-item-{idx:02d}",
            dataset_name=DATASET_NAME,
            input={"query": item["query"]},
            expected_output=item["expected_answer"],
            metadata={
                "expected_chunk_ids": item["expected_chunk_ids"],
                "category": item["category"]
            }
        )
    langfuse.flush()
    print(f"Đã lưu thành công {len(EVAL_ITEMS)} items vào dataset '{DATASET_NAME}'.")


if __name__ == "__main__":
    
    main()