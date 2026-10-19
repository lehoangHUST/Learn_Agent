import re
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class UserResume(BaseModel):
    full_name: str = Field(..., description="Họ và tên đầy đủ của ứng viên")
    phone_number: Optional[str] = Field(None, description="Số điện thoại của ứng viên")
    age: Optional[int] = Field(None, description="Tuổi của ứng viên")
    
    # nghiệm vụ 1: Giới hạn độ tuổi sử dụng
    @field_validator("age")
    @classmethod
    def validate_age_range(cls, v: int) -> int:
        if v < 18 or v > 65:
            raise ValueError(f"Độ tuổi {v} không nằm trong độ tuổi hợp lệ (18 - 65).")
        return v
    
    # Nghiệp vụ 2: Số điện thoại phải đúng định dạng viễn thông Việt Nam
    @field_validator
    @classmethod
    def validate_vietnam_phone(cls, v: str) -> str:
        # Chuẩn hóa: xóa khoảng trắng, gạch nối
        clean_phone = re.sub(r"[\s.-]", "", v)
        pattern = r"^(0|\+84)(3|5|7|8|9)[0-9]{8}$"
        if not re.match(pattern, clean_phone):
            raise ValueError(f"Số điện thoại '{v}' không đúng định dạng nhà mạng Việt Nam.")
        return clean_phone
    
    
structured_llm = llm