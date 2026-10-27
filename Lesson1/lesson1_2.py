from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    FewShotChatMessagePromptTemplate,
)


def get_summary_prompt(
    language: str = "Tiếng Việt",
    length: str = "2 câu",
    use_few_shot: bool = False,
):
    # 1. Sửa lỗi tuple: bỏ các dấu phẩy để tạo chuỗi str hợp lệ
    system_text = (
        "Bạn là một trợ lý AI giúp tóm tắt văn bản.\n"
        "- Ngôn ngữ: {language}\n"
        "- Độ dài: {length}\n"
    )

    blocks = [("system", system_text)]

    if use_few_shot:
        # Cú pháp escape JSON {{ }} hợp lệ
        example_template = ChatPromptTemplate.from_messages([
            ("human", "Tóm tắt: {input_text}"),
            ("ai", '{{"summary": "{sample_summary}"}}'),
        ])

        examples = [
            {
                "input_text": "Hôm nay trời mưa to, đường ngập nặng khiến giao thông tê liệt.",
                "sample_summary": "Trời mưa lớn gây ngập lụt và tắc đường nghiêm trọng.",
            }
        ]

        few_shot = FewShotChatMessagePromptTemplate(
            example_prompt=example_template,
            examples=examples,
        )
        blocks.append(few_shot)

    blocks.append(MessagesPlaceholder("history", optional=True))
    blocks.append(("human", "Hãy tóm tắt bài này:\n{text}"))

    prompt = ChatPromptTemplate.from_messages(blocks)
    return prompt.partial(language=language, length=length)


# =====================================================================
# UNIT TESTS (Kiểm thử cấu trúc message sinh ra mà không cần gọi LLM)
# =====================================================================
def run_unit_tests():
    print("--- Đang chạy Unit Tests ---")

    # TEST 1: Kiểm tra trường hợp KHÔNG có few-shot
    prompt_fast = get_summary_prompt(language="Tiếng Anh", length="1 câu", use_few_shot=False)
    val_fast = prompt_fast.invoke({"text": "Nội dung bài viết mẫu"})
    msgs_fast = val_fast.to_messages()

    assert len(msgs_fast) == 2, f"Kỳ vọng 2 messages nhưng nhận được {len(msgs_fast)}"
    assert msgs_fast[0].type == "system", "Message 0 phải là system"
    assert msgs_fast[1].type == "human", "Message 1 phải là human"
    assert "Tiếng Anh" in msgs_fast[0].content, "Thiếu tham số language đã partial"
    assert "1 câu" in msgs_fast[0].content, "Thiếu tham số length đã partial"
    assert "Nội dung bài viết mẫu" in msgs_fast[1].content
    print("[PASS] Test 1: Cấu trúc prompt không có few-shot chính xác.")

    # TEST 2: Kiểm tra trường hợp CÓ few-shot và định dạng JSON mẫu
    prompt_fs = get_summary_prompt(use_few_shot=True)
    val_fs = prompt_fs.invoke({"text": "Tin tức kinh tế"})
    msgs_fs = val_fs.to_messages()

    # 1 system + 1 human ví dụ + 1 ai ví dụ + 1 human thực tế = 4 messages
    assert len(msgs_fs) == 4, f"Kỳ vọng 4 messages nhưng nhận được {len(msgs_fs)}"
    assert msgs_fs[2].type == "ai"
    # Kiểm tra JSON mẫu trong message của AI không bị lỗi định dạng
    assert '{"summary":' in msgs_fs[2].content, "JSON mẫu của few-shot bị parse sai định dạng"
    print("[PASS] Test 2: Few-shot và JSON mẫu hoạt động chuẩn xác.")

    # TEST 3: Kiểm tra thiếu biến bắt buộc thì phải báo lỗi rõ ràng
    prompt_validate = get_summary_prompt()
    try:
        # Cố tình không truyền biến 'text'
        prompt_validate.invoke({})
        assert False, "Lẽ ra phải ném lỗi khi thiếu biến 'text'!"
    except KeyError as e:
        print(f"[PASS] Test 3: Thiếu biến đã bắt lỗi rõ ràng (KeyError: {e}).")

    print("\n TOÀN BỘ TIÊU CHÍ ĐÃ ĐẠT!")


if __name__ == "__main__":
    run_unit_tests()