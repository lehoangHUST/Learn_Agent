"""State schema, context schema và các hằng số intent của graph.

Ba lớp dữ liệu, ba chỗ khác nhau
--------------------------------
=================  ==========================================  ===================
Loại               Ở đâu                                        Có checkpoint?
=================  ==========================================  ===================
Thay đổi khi chạy  ``FaceAgentState``                            Có
Chỉ đọc, 1 lần run ``FaceAgentContext`` (``context_schema``)     Không
Hạ tầng/dependency ``FaceAgentContext.deps``                     Không
=================  ==========================================  ===================

Vì sao bytes ảnh và embedding **không** nằm trong state:

* Checkpointer serialize state sau mỗi super-step. Một ảnh 5 MB và vài vector 512-D
  sẽ được ghi lại nhiều lần trong một lần chạy - vừa chậm vừa phình.
* Ảnh mặt người và embedding là dữ liệu sinh trắc học. Thứ nằm trong checkpoint là
  thứ tồn tại lâu dài trên đĩa; ta không muốn nó ở đó.
* State phải serialize được. ``ValidatedImage`` chứa ``bytes``, ``np.ndarray`` thì
  không phải JSON.

Nên: ảnh sống trong ``ImageBuffer`` thuộc context (chỉ trong bộ nhớ, chỉ trong một
lần chạy), state chỉ mang ``image_refs``. Embedding sống trong result store có TTL,
state chỉ mang ``detection_result_id``.

Reducer
-------
* ``messages`` -> ``add_messages``: nối, upsert theo id, chuẩn hoá dict/tuple.
* ``errors`` / ``trace`` / ``executed_steps`` -> ``append_run_scoped``: tích lũy
  trong phạm vi một lần chạy. Đây là các field "lịch sử": một node chỉ nói thêm phần
  của nó, không bao giờ ghi đè phần node khác đã ghi. Nhờ vậy nếu sau này có nhánh
  chạy song song, chúng vẫn gộp đúng mà không cần sửa node nào.
* ``warnings`` -> ``dedup_append``: nối, bỏ trùng, cùng ngữ nghĩa reset.

Lưu ý quan trọng về tích luỹ + checkpoint: state được nạp lại theo ``thread_id``,
nên một channel nối thuần sẽ mang theo dữ liệu của **mọi lượt trước** trong cùng
cuộc hội thoại. Điều đó đúng cho ``messages`` và sai cho mọi field mô tả một lần
chạy. Hai reducer trên vì vậy coi ``update is None`` là "lượt mới, xoá sạch", và
``FaceAgentRunner.run`` gửi tín hiệu đó ở mỗi lượt.
* Các field còn lại -> reducer mặc định (ghi đè), vì chúng mang nghĩa "trạng thái
  hiện tại", chỉ có một giá trị đúng tại một thời điểm.

Hai node song song cùng ghi một field *ghi đè* sẽ gây ``InvalidUpdateError``. Graph
này được thiết kế tuần tự nên điều đó không xảy ra; nhưng cách chia field ở trên là
thứ giữ cho nó vẫn đúng nếu về sau có fan-out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, NotRequired

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.security import Principal
from app.vision.preprocessing import ValidatedImage

if TYPE_CHECKING:  # tránh import vòng; chỉ cần cho type hint
    from app.dependencies import AgentDeps

# --------------------------------------------------------------------------- #
# Intent
# --------------------------------------------------------------------------- #
INTENT_FACE_DETECTION = "face_detection"
INTENT_FACE_SEARCH = "face_search"
INTENT_DETECTION_AND_SEARCH = "face_detection_and_search"
INTENT_RESULT_QUESTION = "result_question"
INTENT_GENERAL_FACE_QUESTION = "general_face_question"
INTENT_UNSUPPORTED = "unsupported_request"

ALL_INTENTS = (
    INTENT_FACE_DETECTION,
    INTENT_FACE_SEARCH,
    INTENT_DETECTION_AND_SEARCH,
    INTENT_RESULT_QUESTION,
    INTENT_GENERAL_FACE_QUESTION,
    INTENT_UNSUPPORTED,
)

# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"


def append_run_scoped(current: list[Any] | None, update: list[Any] | None) -> list[Any]:
    """Nối, nhưng ``update is None`` nghĩa là "bắt đầu lượt mới" -> xoá sạch.

    Vì sao cần ngữ nghĩa reset
    --------------------------
    Checkpoint của LangGraph sống theo ``thread_id``, nên state của lượt trước được
    nạp lại ở lượt sau. Với một channel ``operator.add`` thuần, ``executed_steps``
    của lượt 2 sẽ là các bước của lượt 1 **cộng** lượt 2, ``errors`` của lượt 1 vẫn
    còn đó ở lượt 3. Đó là hành vi đúng cho ``messages`` (lịch sử hội thoại phải tích
    luỹ) nhưng **sai** cho các field mô tả *một lần chạy*.

    Và không thể reset bằng cách gửi ``[]``: với phép nối thì ``[]`` không thêm gì,
    không xoá gì. Nên cần một tín hiệu ngoài miền giá trị, ở đây là ``None``, và
    runner gửi nó trong input ở mỗi lượt (xem ``FaceAgentRunner.run``).

    Vẫn giữ các tính chất của một reducer tốt trong phạm vi một lần chạy: thuần,
    chịu được ``None`` ở giá trị cũ, và kết hợp được.
    """
    if update is None:
        return []
    return list(current or []) + list(update)


def dedup_append(current: list[str] | None, update: list[str] | None) -> list[str]:
    """Như ``append_run_scoped`` nhưng bỏ trùng - dùng cho ``warnings``.

    Vì sao không dùng ``operator.add``: nhiều node cùng báo một cảnh báo là chuyện
    bình thường ở đây - "ngưỡng chưa hiệu chuẩn" được phát ra bởi cả search service,
    analyze và answer. Với ``operator.add`` thì người dùng thấy nó ba lần. Dedup
    thuộc về *cách gộp của channel*, nên nó là việc của reducer, không phải việc
    của từng node (nếu để node tự kiểm tra trùng thì mỗi node phải đọc giá trị cũ -
    đúng thứ mà partial update sinh ra để tránh).

    Tính chất: thuần, chịu được ``None`` ở giá trị cũ, kết hợp được, và idempotent
    với bản ghi trùng - nên vẫn đúng nếu về sau có nhánh chạy song song.
    """
    if update is None:
        return []
    out = list(current or [])
    seen = set(out)
    for item in update:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


class ImageRef(TypedDict):
    """Tham chiếu tới một ảnh trong ``ImageBuffer`` của lần chạy này.

    Chứa đủ dữ kiện để trả lời câu hỏi về ảnh (kích thước, định dạng) mà không
    chứa một byte nội dung nào.
    """

    ref: str
    sha256: str
    mime: str
    width: int
    height: int
    size_bytes: int


class ErrorEntry(TypedDict):
    code: str
    message: str
    retryable: bool
    node: str
    details: NotRequired[dict[str, Any]]


class PlanStep(TypedDict):
    step: str
    reason: str


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
class FaceAgentInput(TypedDict):
    """Hợp đồng ĐẦU VÀO của graph.

    Hẹp hơn state nội bộ một cách có chủ ý: người gọi không set được
    ``intent``, ``status``, ``answer`` hay bất kỳ field nội bộ nào. Nếu dùng chung
    một schema cho cả input và state, một client có thể tự đặt
    ``search_result_id`` của người khác và graph sẽ tin.
    """

    query: str
    request_id: str
    thread_id: str
    user_id: str
    image_refs: NotRequired[list[ImageRef]]
    requested_face_indices: NotRequired[list[int] | None]
    target_result_id: NotRequired[str | None]
    messages: NotRequired[Annotated[list[AnyMessage], add_messages]]


class FaceAgentOutput(TypedDict):
    """Hợp đồng ĐẦU RA. Những gì không có ở đây vẫn sống trong state/checkpoint,
    chỉ là không trả ra ngoài."""

    request_id: str
    thread_id: str
    status: str
    intent: str
    intent_source: str
    answer: str
    answer_source: str
    detection_result_id: str | None
    search_result_id: str | None
    detection_summary: dict[str, Any] | None
    search_summary: dict[str, Any] | None
    analysis: dict[str, Any] | None
    errors: list[ErrorEntry]
    warnings: list[str]
    executed_steps: list[str]


class FaceAgentState(TypedDict):
    """State nội bộ đầy đủ của graph."""

    # ---- định danh ----
    request_id: str
    thread_id: str
    user_id: str

    # ---- đầu vào ----
    query: str
    image_refs: list[ImageRef]
    requested_face_indices: list[int] | None
    target_result_id: str | None

    # ---- hội thoại ----
    messages: Annotated[list[AnyMessage], add_messages]

    # ---- điều phối ----
    # capabilities: những gì lượt này làm được (có ảnh mới? có kết quả cũ? index có
    # dữ liệu?). Do validate_input tính, do classify_intent và plan_execution đọc.
    # Tách riêng khỏi `grounding` để node answer không bị phụ thuộc vào thứ tự ghi
    # của hai node khác - cả hai field đều dùng reducer ghi đè.
    capabilities: dict[str, Any] | None
    intent: str
    intent_source: str  # "rule" | "llm" | "fallback"
    execution_plan: list[PlanStep]
    status: str
    retry_count: int
    # Node mà handle_error quyết định chạy lại. None = không retry nữa.
    retry_target: str | None

    # ---- kết quả (chỉ id + payload công khai, không embedding) ----
    detection_result_id: str | None
    detection_summary: dict[str, Any] | None
    search_result_id: str | None
    search_summary: dict[str, Any] | None

    # ---- tham chiếu embedding ----
    # Vector KHÔNG đi qua state (không serialize được, lại là dữ liệu sinh trắc học).
    # State chỉ mang: kết quả nào đang giữ embedding, và những face nào sẽ được truy
    # vấn. Node search đọc lại vector từ result store theo hai field này - một lần
    # đọc SQLite cục bộ, rẻ hơn nhiều so với việc nhồi ma trận vào checkpoint.
    embedding_source_result_id: str | None
    query_face_indices: list[int] | None

    # ---- nội bộ: không có trong input/output schema ----
    # grounding là tập dữ kiện mà node answer được phép dùng. Nó tồn tại để câu trả
    # lời bị *giới hạn* trong dữ liệu thật: prompt chỉ nhận field này.
    grounding: dict[str, Any] | None
    analysis: dict[str, Any] | None
    answer: str
    # "template" (sinh tất định từ grounding) hoặc "llm" (do model diễn đạt lại).
    # Trả ra ngoài vì người đọc có quyền biết phần văn bản nào do model sinh ra.
    answer_source: str

    # ---- field tích lũy, phạm vi MỘT lần chạy (xem docstring module) ----
    # Dùng reducer có ngữ nghĩa reset: runner gửi None ở đầu mỗi lượt để xoá dữ liệu
    # của lượt trước đã được checkpoint nạp lại.
    errors: Annotated[list[ErrorEntry], append_run_scoped]
    warnings: Annotated[list[str], dedup_append]
    trace: Annotated[list[str], append_run_scoped]
    executed_steps: Annotated[list[str], append_run_scoped]

    # Tín hiệu "node vừa chạy có lỗi không" - do decorator graph_node ghi, do các
    # router đọc. Đây là channel ghi đè, không tích luỹ: nó nói về *bước hiện tại*.
    # Vì sao không để router tự suy từ errors[-1]: sau khi retry thành công, lỗi cũ
    # VẪN còn trong errors (đó là lịch sử), nên errors[-1].node vẫn trỏ vào node vừa
    # chạy thành công -> router lại đẩy về handle_error, thành vòng lặp.
    failed_node: str | None


class AnswerNodeInput(TypedDict):
    """Schema hẹp cho node ``answer_question``.

    Node này chỉ được *nhìn thấy* phần nó cần. Không phải chuyện thẩm mỹ: nó là node
    duy nhất gọi LLM, nên thu hẹp tầm nhìn của nó là thu hẹp đúng chỗ dữ liệu có thể
    rò ra prompt.
    """

    query: str
    intent: str
    grounding: dict[str, Any] | None
    analysis: dict[str, Any] | None
    # Phải khai báo cùng reducer như trong FaceAgentState: channel được xác định
    # theo TÊN, nên hai schema nói khác nhau về cùng một channel là xung đột.
    errors: Annotated[list[ErrorEntry], append_run_scoped]
    warnings: Annotated[list[str], dedup_append]
    messages: Annotated[list[AnyMessage], add_messages]


# --------------------------------------------------------------------------- #
# Context (chỉ đọc trong một lần chạy)
# --------------------------------------------------------------------------- #
class ImageBuffer:
    """Giữ ảnh của một lần chạy trong RAM, truy cập theo ref.

    Không ghi đĩa, không nằm trong checkpoint, mất khi request kết thúc. Đây là lý
    do state chỉ cần mang ``image_refs``.
    """

    __slots__ = ("_images",)

    def __init__(self, images: dict[str, ValidatedImage] | None = None) -> None:
        self._images: dict[str, ValidatedImage] = dict(images or {})

    def add(self, ref: str, image: ValidatedImage) -> ImageRef:
        self._images[ref] = image
        return ImageRef(
            ref=ref,
            sha256=image.sha256,
            mime=image.mime,
            width=image.width,
            height=image.height,
            size_bytes=image.size_bytes,
        )

    def get(self, ref: str) -> ValidatedImage | None:
        return self._images.get(ref)

    def __contains__(self, ref: object) -> bool:
        return ref in self._images

    def __len__(self) -> int:
        return len(self._images)


@dataclass(slots=True)
class FaceAgentContext:
    """``context_schema`` của graph: dữ liệu chỉ đọc + dependency.

    Vì sao không nhét vào state: principal không đổi trong một lần chạy, service là
    object không serialize được, và ``ImageBuffer`` chứa bytes ảnh. Cả ba đều không
    nên đi vào checkpoint.
    """

    principal: Principal
    collection: str
    deps: AgentDeps
    images: ImageBuffer = field(default_factory=ImageBuffer)


def run_scoped_reset() -> dict[str, Any]:
    """State cần xoá ở đầu mỗi lượt để dữ liệu lượt trước không rò sang.

    Checkpoint nạp lại state theo ``thread_id``, nên nếu không reset thì lượt 2 sẽ
    thấy ``executed_steps``, ``errors``, ``intent``, ``detection_summary``... của lượt 1.
    Những field tích luỹ nhận ``None`` (tín hiệu reset của ``append_run_scoped``);
    những field ghi đè nhận giá trị rỗng tường minh.

    ``messages`` **không** nằm ở đây: lịch sử hội thoại tích luỹ là đúng.

    Dict này được node ``begin_run`` trả về, **không** gửi qua input của graph: với
    ``input_schema`` hẹp, LangGraph lọc bỏ im lặng mọi key không thuộc schema đó, nên
    reset gửi từ ngoài sẽ không có tác dụng nào (và không báo lỗi gì - đúng loại bug
    chỉ lộ ra ở lượt thứ hai).
    """
    return {
        # tích luỹ -> None để xoá
        "errors": None,
        "warnings": None,
        "trace": None,
        "executed_steps": None,
        # ghi đè -> đặt lại tường minh
        "failed_node": None,
        "intent": "",
        "intent_source": "rule",
        "capabilities": None,
        "execution_plan": [],
        "status": STATUS_PENDING,
        "retry_count": 0,
        "retry_target": None,
        "detection_result_id": None,
        "detection_summary": None,
        "search_result_id": None,
        "search_summary": None,
        "embedding_source_result_id": None,
        "query_face_indices": None,
        "grounding": None,
        "analysis": None,
        "answer": "",
        "answer_source": "template",
    }


def initial_state(
    *,
    query: str,
    request_id: str,
    thread_id: str,
    user_id: str,
    image_refs: list[ImageRef] | None = None,
    requested_face_indices: list[int] | None = None,
    target_result_id: str | None = None,
) -> FaceAgentInput:
    """Dựng input cho graph. Dùng chung bởi API và test để không lệch nhau.

    Chỉ chứa các field của ``FaceAgentInput``. Việc dọn state lượt trước là của
    node ``begin_run``, không phải của input - xem ``run_scoped_reset``.
    """
    return {
        "query": query,
        "request_id": request_id,
        "thread_id": thread_id,
        "user_id": user_id,
        "image_refs": image_refs or [],
        "requested_face_indices": requested_face_indices,
        "target_result_id": target_result_id,
    }
