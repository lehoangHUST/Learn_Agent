"""Node ``classify_intent``: rule trước, LLM chỉ khi rule không chắc.

Vì sao không đưa hết cho LLM
----------------------------
Phân loại intent ở đây phần lớn là việc xác định rõ: có ảnh mới hay không, câu có
chứa từ khoá tìm kiếm hay đếm. Dùng LLM cho việc đó đổi một hàm tất định thành một
lời gọi mạng có thể lỗi, chậm hơn hai bậc, tốn tiền, và cho kết quả khác nhau giữa
các lần chạy. Yêu cầu của hệ thống cũng nói thẳng: "không yêu cầu LLM lập kế hoạch
phức tạp cho các tác vụ xác định rõ".

Nên thứ tự là: ``classify_by_rules`` chạy trước và tự khai báo mình có **chắc**
hay không. Chỉ khi không chắc (và LLM được bật) thì mới gọi LLM. LLM lỗi -> giữ
nguyên kết quả của rule, ghi một warning. Không có đường nào làm sập request.

``classify_by_rules`` là hàm thuần, không I/O, không phụ thuộc graph -> test trực
tiếp với hàng chục câu đầu vào mà không cần dựng gì.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from langgraph.runtime import Runtime

from app.agent.nodes.common import graph_node
from app.agent.state import (
    INTENT_DETECTION_AND_SEARCH,
    INTENT_FACE_DETECTION,
    INTENT_FACE_SEARCH,
    INTENT_GENERAL_FACE_QUESTION,
    INTENT_RESULT_QUESTION,
    INTENT_UNSUPPORTED,
    FaceAgentContext,
    FaceAgentState,
)
from app.errors import LLMError

logger = logging.getLogger(__name__)


#   NFD tách được dấu của ă â ê ô ơ ư và các dấu thanh, nhưng KHÔNG tách được
#   "đ" (U+0111): đó là một ký tự riêng, không phải "d" + dấu, nên nó không có dạng
#   phân rã. Thiếu bước map này thì mọi từ khoá có "đ" - "đếm", "đối chiếu",
#   "độ tuổi", "đầu tiên" - sẽ không bao giờ khớp, vì danh sách từ khoá viết không
#   dấu là "dem", "doi chieu", "do tuoi", "dau tien".
_SPECIAL_LETTERS = str.maketrans({"đ": "d", "Đ": "D"})


def strip_accents(text: str) -> str:
    """Bỏ dấu để so khớp từ khoá không phụ thuộc cách người dùng gõ."""
    decomposed = unicodedata.normalize("NFD", (text or "").translate(_SPECIAL_LETTERS))
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", strip_accents(text or "").lower()).strip()


# Yêu cầu hệ thống từ chối: suy đoán thuộc tính con người, hoặc theo dõi người.
UNSUPPORTED_KEYWORDS = (
    "tuoi",
    "do tuoi",
    "gioi tinh",
    "nam hay nu",
    "cam xuc",
    "bieu cam",
    "chung toc",
    "dan toc",
    "theo doi",
    "truy vet",
    "giam sat",
    "age",
    "gender",
    "emotion",
    "ethnicity",
    "race",
    "track this person",
    "surveillance",
)

SEARCH_KEYWORDS = (
    "tim kiem",
    "tim",
    "nhan dang",
    "la ai",
    "ai day",
    "ai trong",
    "danh tinh",
    "so khop",
    "doi chieu",
    "tra cuu",
    "gan nhat",
    "search",
    "identify",
    "match",
    "lookup",
    "top",
)

DETECT_KEYWORDS = (
    "phat hien",
    "bao nhieu khuon mat",
    "bao nhieu mat",
    "bao nhieu guong mat",
    "co may khuon mat",
    "may khuon mat",
    "dem khuon mat",
    "dem mat",
    "detect",
    "bounding box",
    "landmark",
    "toa do",
    "vi tri khuon mat",
)

RESULT_KEYWORDS = (
    "ket qua",
    "similarity",
    "diem giong",
    "thu hang",
    "xep hang",
    "so sanh",
    "vi sao",
    "tai sao",
    "nguong",
    "threshold",
    "tom tat",
    "giai thich",
    "confidence",
    "khong tim thay",
    "khong co ket qua",
)

GENERAL_KEYWORDS = (
    "la gi",
    "khac nhau",
    "khac biet",
    "nghia la",
    "the nao",
    "embedding",
    "verification",
    "identification",
    "co che",
    "hoat dong",
)

_ORDINALS = {
    "nhat": 0,
    "dau": 0,
    "dau tien": 0,
    "hai": 1,
    "ba": 2,
    "tu": 3,
    "bon": 3,
    "nam": 4,
    "sau": 5,
    "bay": 6,
    "tam": 7,
    "chin": 8,
    "muoi": 9,
}

_FACE_WORDS = r"(?:khuon mat|guong mat|mat|face)"


@dataclass(slots=True)
class RuleIntent:
    intent: str
    confident: bool
    reason: str
    face_indices: list[int] | None = None
    signals: dict[str, Any] = field(default_factory=dict)


def extract_face_indices(normalized: str) -> tuple[list[int] | None, str]:
    """Tìm chỉ số khuôn mặt mà người dùng nhắc tới (0-based).

    Chỉ nhận ordinal **gắn với một từ chỉ khuôn mặt**. Phân biệt này quan trọng:
    "khuôn mặt thứ hai" nói về face_index 1, còn "kết quả thứ hai" nói về ứng viên
    xếp thứ 2 trong kết quả search - hai thứ hoàn toàn khác nhau. Gộp chúng lại là
    tìm sai khuôn mặt.
    """
    # face_index=N -> đã là 0-based, dùng nguyên
    explicit = re.search(r"face_index\s*[=:]\s*(\d+)", normalized)
    if explicit:
        return [int(explicit.group(1))], "face_index tường minh trong câu"

    # "khuôn mặt thứ hai" / "face thu 2"
    ordinal_word = re.search(rf"{_FACE_WORDS}\s+thu\s+([a-z]+|\d+)", normalized)
    if ordinal_word:
        token = ordinal_word.group(1)
        if token.isdigit():
            value = int(token)
            if value >= 1:
                return [value - 1], f"'thứ {value}' -> face_index {value - 1}"
        elif token in _ORDINALS:
            return [_ORDINALS[token]], f"'thứ {token}' -> face_index {_ORDINALS[token]}"

    # "mặt đầu tiên" - không có chữ "thứ". Phải khớp tường minh chứ không thể bỏ
    # "thu" thành tuỳ chọn trong regex ordinal chung: khi đó "mặt nam" sẽ bị hiểu là
    # "mặt thứ năm" (nam = 4 trong bảng ordinal).
    if re.search(rf"{_FACE_WORDS}\s+dau(\s+tien)?\b", normalized):
        return [0], "'đầu tiên' -> face_index 0"

    # "khuôn mặt 2" / "mặt số 3" -> hiểu là 1-based
    numbered = re.search(rf"{_FACE_WORDS}\s+(?:so\s+)?(\d+)", normalized)
    if numbered:
        value = int(numbered.group(1))
        if value >= 1:
            return [value - 1], f"'khuôn mặt {value}' -> face_index {value - 1}"

    # "mặt to nhất": các mặt được sắp theo diện tích giảm dần nên index 0 là lớn nhất.
    if re.search(rf"{_FACE_WORDS}\s+(?:to|lon)\s+nhat", normalized) or re.search(
        r"(?:to|lon)\s+nhat", normalized
    ):
        return [0], "các khuôn mặt được sắp theo diện tích giảm dần nên index 0 là lớn nhất"

    return None, ""


def _hits(normalized: str, keywords: tuple[str, ...]) -> list[str]:
    """So khớp theo ranh giới từ.

    Dùng ``\\b`` thay vì ``in`` để "tim" không khớp trong "timer" và để từ khoá ở
    cuối câu (không có dấu cách theo sau) vẫn được nhận.
    """
    found: list[str] = []
    for kw in keywords:
        kw = kw.strip()
        if kw and re.search(rf"\b{re.escape(kw)}\b", normalized):
            found.append(kw)
    return found


def classify_by_rules(
    query: str,
    *,
    has_new_images: bool,
    prior_detection: bool,
    prior_search: bool,
) -> RuleIntent:
    """Phân loại tất định. ``confident=False`` nghĩa là "nên hỏi LLM nếu có"."""
    norm = normalize(query)
    face_indices, index_reason = extract_face_indices(norm)

    unsupported = _hits(norm, UNSUPPORTED_KEYWORDS)
    search = _hits(norm, SEARCH_KEYWORDS)
    detect = _hits(norm, DETECT_KEYWORDS)
    result_q = _hits(norm, RESULT_KEYWORDS)
    general = _hits(norm, GENERAL_KEYWORDS)

    signals = {
        "normalized_query": norm,
        "unsupported_hits": unsupported,
        "search_hits": search,
        "detect_hits": detect,
        "result_hits": result_q,
        "general_hits": general,
        "has_new_images": has_new_images,
        "prior_detection": prior_detection,
        "prior_search": prior_search,
        "face_index_reason": index_reason,
    }

    def out(intent: str, confident: bool, reason: str) -> RuleIntent:
        return RuleIntent(
            intent=intent,
            confident=confident,
            reason=reason,
            face_indices=face_indices,
            signals=signals,
        )

    # 1. Từ chối trước mọi thứ khác: không bao giờ chạy model cho yêu cầu này.
    if unsupported:
        return out(
            INTENT_UNSUPPORTED,
            True,
            f"câu hỏi yêu cầu thuộc tính/hành vi hệ thống không hỗ trợ: {unsupported}",
        )

    if has_new_images:
        if search and detect:
            return out(
                INTENT_DETECTION_AND_SEARCH,
                True,
                "câu hỏi yêu cầu cả phát hiện và tìm kiếm",
            )
        if search:
            return out(INTENT_FACE_SEARCH, True, f"từ khoá tìm kiếm: {search}")
        if detect:
            return out(INTENT_FACE_DETECTION, True, f"từ khoá phát hiện: {detect}")
        # Có ảnh nhưng câu không rõ -> chọn việc rẻ nhất mà vẫn có ích, và để ngỏ
        # cho LLM xem lại.
        return out(
            INTENT_FACE_DETECTION,
            False,
            "có ảnh mới nhưng câu hỏi không nêu rõ việc cần làm; mặc định phát hiện "
            "khuôn mặt (rẻ hơn tìm kiếm) và chờ LLM xác nhận nếu được bật",
        )

    # Không có ảnh mới.
    # Hỏi-về-kết-quả được xét TRƯỚC tìm-kiếm. Lý do: câu như "tại sao không tìm thấy
    # kết quả phù hợp?" chứa cả từ khoá tìm kiếm ("tìm") lẫn từ khoá kết quả, nhưng ý
    # người dùng là hỏi về lần tìm đã chạy, không phải yêu cầu chạy lại. Chọn nhánh
    # này cũng là chọn việc không tốn tài nguyên.
    if result_q and (prior_detection or prior_search):
        return out(
            INTENT_RESULT_QUESTION,
            # Có cả hai loại từ khoá -> không chắc, để LLM xem lại nếu được bật.
            not search,
            f"hỏi về kết quả đã có: {result_q}"
            + (f" (câu cũng có từ khoá tìm kiếm {search}, chọn nhánh rẻ hơn)" if search else ""),
        )

    if search:
        if prior_detection:
            return out(
                INTENT_FACE_SEARCH,
                True,
                "không có ảnh mới nhưng thread đã có kết quả detection -> tìm kiếm "
                "bằng embedding đã lưu",
            )
        return out(
            INTENT_FACE_SEARCH,
            True,
            "yêu cầu tìm kiếm nhưng không có ảnh mới và cũng chưa có kết quả "
            "detection nào trong thread",
        )

    # Câu hỏi rõ ràng nói về kết quả nhưng thread không có kết quả nào. Phải vẫn là
    # result_question để plan_execution báo "chưa có kết quả nào, hãy gửi ảnh".
    # Nếu rơi xuống general_face_question thì người dùng nhận một bài giải thích khái
    # niệm khi họ hỏi "kết quả thứ hai similarity bao nhiêu" - trả lời lạc đề và tệ
    # hơn: nó che mất việc dữ liệu đã hết hạn hoặc không thuộc về họ.
    if result_q and not general:
        return out(
            INTENT_RESULT_QUESTION,
            True,
            f"hỏi về kết quả ({result_q}) nhưng thread chưa có kết quả nào còn hiệu lực",
        )

    if general:
        return out(INTENT_GENERAL_FACE_QUESTION, True, f"hỏi kiến thức chung: {general}")

    if prior_detection or prior_search:
        return out(
            INTENT_RESULT_QUESTION,
            False,
            "không có ảnh mới, thread đã có kết quả -> đoán là hỏi về kết quả cũ",
        )

    return out(
        INTENT_GENERAL_FACE_QUESTION,
        False,
        "không có ảnh, không có kết quả cũ, không có từ khoá rõ ràng",
    )


@graph_node("classify_intent")
def classify_intent(state: FaceAgentState, runtime: Runtime[FaceAgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    caps = state.get("capabilities") or {}

    decision = classify_by_rules(
        state.get("query", ""),
        has_new_images=bool(caps.get("has_new_images")),
        prior_detection=bool(caps.get("prior_detection_result_id")),
        prior_search=bool(caps.get("prior_search_result_id")),
    )

    intent = decision.intent
    source = "rule"
    reason = decision.reason
    face_indices = decision.face_indices
    warnings: list[str] = []

    llm = ctx.deps.llm
    use_llm = (
        not decision.confident
        and llm is not None
        and ctx.deps.settings.llm.use_for_intent
    )
    if use_llm:
        try:
            llm_decision = llm.classify_intent(
                state.get("query", ""),
                has_new_images=bool(caps.get("has_new_images")),
                prior_detection=bool(caps.get("prior_detection_result_id")),
                prior_search=bool(caps.get("prior_search_result_id")),
            )
            intent = llm_decision.intent
            source = "llm"
            reason = llm_decision.reason or "LLM phân loại"
            # Rule đã trích được index cụ thể thì giữ nguyên: nó dựa trên regex xác
            # định, đáng tin hơn suy luận của model cho loại thông tin này.
            if face_indices is None and llm_decision.face_indices:
                face_indices = [i for i in llm_decision.face_indices if i >= 0] or None
        except LLMError as exc:
            source = "rule"
            warnings.append(
                f"LLM phân loại intent thất bại ({exc.code}), dùng kết quả rule: {decision.intent}"
            )
            logger.warning(
                "intent_llm_failed",
                extra={"error_code": exc.code, "fallback_intent": decision.intent},
            )

    # Người dùng chỉ rõ face_index thì nó thắng kết quả suy luận.
    explicit = state.get("requested_face_indices")
    if explicit:
        face_indices = list(explicit)

    return {
        "intent": intent,
        "intent_source": source,
        "requested_face_indices": face_indices,
        "capabilities": {
            **caps,
            "intent_reason": reason,
            "intent_rule_confident": decision.confident,
            "intent_signals": decision.signals,
        },
        "warnings": warnings,
    }
