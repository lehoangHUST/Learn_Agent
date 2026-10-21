import os
import re
import pytest
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from langchain.chat_models import init_chat_model
from langchain_community.chat_models import FakeListChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException


os.environ.pop("SSL_CERT_FILE", None)
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError(
        "[LỖI THIẾU KEY] Chưa tìm thấy biến môi trường 'GROQ_API_KEY'. "
        "Vui lòng khai báo trong file .env hoặc export vào terminal."
    )
    

# init chat model
chat = init_chat_model("groq:openai/gpt-oss-20b")

class UserResume(BaseModel):
    full_name: str = Field(..., description="Họ và tên đầy đủ của ứng viên")
    phone_number: Optional[str] = Field(description="Số điện thoại của ứng viên")
    age: Optional[int] = Field(None, description="Tuổi của ứng viên")
    
    # nghiệm vụ 1: Giới hạn độ tuổi sử dụng
    @field_validator("age")
    @classmethod
    def validate_age_range(cls, v: int) -> int:
        if v < 18 or v > 65:
            raise ValueError(f"Độ tuổi {v} không nằm trong độ tuổi hợp lệ (18 - 65).")
        return v
    
    # Nghiệp vụ 2: Số điện thoại phải đúng định dạng viễn thông Việt Nam
    @field_validator("phone_number")
    @classmethod
    def validate_vietnam_phone(cls, v: str) -> str:
        # Chuẩn hóa: xóa khoảng trắng, gạch nối
        clean_phone = re.sub(r"[\s.-]", "", v)
        pattern = r"^(0|\+84)(3|5|7|8|9)[0-9]{8}$"
        if not re.match(pattern, clean_phone):
            raise ValueError(f"Số điện thoại '{v}' không đúng định dạng nhà mạng Việt Nam.")
        return clean_phone
    
parser = PydanticOutputParser(pydantic_object=UserResume)


## TEST 1: Output valid
def test_valid_output():
    fake_llm = FakeListChatModel(responses=[
        '{"full_name": "Lê Huy Hoàng", "age": 35, "phone_number": "0775172945"}'
    ])

    chain = fake_llm | parser
    result = chain.invoke("Input bất kỳ vì fake model không quan tâm input")
    
    print(result)
    # Kiểm tra kết quả parse ra đúng kiểu và giá trị
    assert isinstance(result, UserResume)
    assert result.full_name == "Lê Huy Hoàng"
    assert result.age == 35
    assert result.phone_number == "0775172945"
    

## TEST 2: Output invalid
def test_invalid_output_raises_error():
    fake_llm = FakeListChatModel(responses=[
        '{"full_name": "Lê Huy Hoàng", "age": 35}',
        'Xin lỗi, tôi không tìm thấy thông tin sản phẩm này.'
    ])
    
    chain = fake_llm | parser
    
    with pytest.raises(OutputParserException):
        chain.invoke("Test sai kiểu dữ liệu")
        
    with pytest.raises(OutputParserException):
        chain.invoke("Test model từ chối trả lời")

## TEST 3: Test
def test_retry_mechanism():
    # Lần 1 trả về sai, Lần 2 (sau khi retry) trả về đúng
    fake_llm = FakeListChatModel(responses=[
        '{"full_name": "Lê Huy Hoàng", "age": 35}',  # Lỗi: thiếu trường bắt buộc 'price'
        '{"full_name": "Lê Huy Hoàng", "age": 35, "phone_number": "0775172945"}'  # Đúng
    ])
    
    chain = fake_llm | parser
    
    # Giả lập logic gọi tối đa 2 lần
    result = None
    for _ in range(2):
        try:
            result = chain.invoke("Trích xuất thông tin người dùng")
            break  # Thành công thì thoát vòng lặp
        except OutputParserException:
            continue  # Lỗi thì thử lại
    
    print(result)        
    assert result is not None
    assert result.full_name == "Lê Huy Hoàng"
    
    
# check result

test_valid_output()
test_invalid_output_raises_error()
test_retry_mechanism()


"""
# 1. Output Parser (Bộ phân tích đầu ra - Xử lý phía Client/Code của bạn)
Output Parser là một đoạn code Python thuần túy chạy trên máy của bạn sau khi LLM đã trả lời xong. Bản thân LLM không hề có tính năng đặc biệt nào can thiệp, nó chỉ đang "chat" bình thường.

# 2. Native Structured Output (Đầu ra có cấu trúc tự nhiên - Xử lý từ gốc ở phía Model/Server)
Chữ Native nghĩa là "tích hợp sẵn từ trong lõi". Thay vì dặn miệng qua prompt rồi cầu nguyện LLM làm đúng, các hãng AI (OpenAI, Google, Anthropic) thiết kế lại hệ thống API và bộ não giải mã của model để ép buộc nó chỉ được phép xuất ra đúng cấu trúc.

"""