"""Node ``validate_input``.

Ảnh đã được validate *trước khi* vào graph (tầng API dựng ``ImageBuffer`` từ các
``ValidatedImage``). Vậy node này còn làm gì?

Nó kiểm tra tính nhất quán của chính *input graph*, vì graph có nhiều đường vào:
API chat, script CLI, test, và sau này có thể là một queue. Những bất biến dưới đây
phải đúng với mọi đường vào, nên chúng được kiểm ở đây chứ không ở tầng HTTP:

* có ``query`` hoặc có ảnh - không thể cả hai đều rỗng,
* mọi ``image_refs`` đều giải được trong ``ImageBuffer`` (ref mồ côi = bug ở caller
  hoặc là ai đó đang thử đoán ref của người khác),
* số ảnh trong giới hạn,
* ``requested_face_indices`` là số nguyên không âm.

Nó cũng xác định *những gì có thể làm được* trong lượt này: có ảnh mới không, thread
này đã có kết quả cũ chưa. Hai dữ kiện đó là đầu vào cho ``classify_intent``, và
quan trọng là chúng được lấy từ **result store có phân quyền**, không phải từ lịch
sử hội thoại.
"""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from app.agent.nodes.common import graph_node
from app.agent.state import (
    STATUS_RUNNING,
    FaceAgentContext,
    FaceAgentState,
)
from app.errors import ValidationFailed
from app.repositories.result_store import KIND_DETECTION, KIND_SEARCH

MAX_QUERY_CHARS = 4000


@graph_node("validate_input")
def validate_input(state: FaceAgentState, runtime: Runtime[FaceAgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    settings = ctx.deps.settings

    query = (state.get("query") or "").strip()
    image_refs = list(state.get("image_refs") or [])

    if not query and not image_refs:
        raise ValidationFailed("request phải có câu hỏi hoặc ít nhất một ảnh")

    if len(query) > MAX_QUERY_CHARS:
        raise ValidationFailed(
            f"câu hỏi dài {len(query)} ký tự, vượt giới hạn {MAX_QUERY_CHARS}",
            details={"length": len(query), "max": MAX_QUERY_CHARS},
        )

    if len(image_refs) > settings.limits.max_images_per_request:
        raise ValidationFailed(
            f"{len(image_refs)} ảnh vượt giới hạn {settings.limits.max_images_per_request}",
            details={"count": len(image_refs), "max": settings.limits.max_images_per_request},
        )

    missing = [ref["ref"] for ref in image_refs if ref["ref"] not in ctx.images]
    if missing:
        # Không nói rõ ref nào tồn tại/không tồn tại ở chỗ khác - chỉ báo là không
        # giải được trong lần chạy này.
        raise ValidationFailed(
            "có image_ref không giải được trong lần chạy này",
            details={"unresolved_count": len(missing)},
        )

    indices = state.get("requested_face_indices")
    # isinstance(i, bool) bị loại riêng: trong Python, True là một int, nên
    # [True, False] sẽ lọt qua một phép kiểm tra isinstance(i, int) đơn thuần và
    # trở thành face_index 1 và 0.
    if indices is not None and (
        not isinstance(indices, list)
        or any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in indices)
    ):
        raise ValidationFailed(
            "requested_face_indices phải là danh sách số nguyên >= 0",
            details={"received": indices},
        )

    # Lịch sử có cấu trúc, lấy qua principal - không đọc từ messages.
    store = ctx.deps.results
    prior_detection = store.latest(
        kind=KIND_DETECTION, principal=ctx.principal, thread_id=state.get("thread_id")
    )
    prior_search = store.latest(
        kind=KIND_SEARCH, principal=ctx.principal, thread_id=state.get("thread_id")
    )

    capabilities = {
        "has_new_images": bool(image_refs),
        "image_count": len(image_refs),
        "prior_detection_result_id": prior_detection.result_id if prior_detection else None,
        "prior_search_result_id": prior_search.result_id if prior_search else None,
        "index_total_faces": ctx.deps.search.stats(ctx.collection)["total_faces"],
    }

    warnings: list[str] = []
    if capabilities["index_total_faces"] == 0:
        warnings.append(
            f"collection {ctx.collection!r} chưa có khuôn mặt nào được đăng ký - "
            "chức năng tìm kiếm sẽ không có ứng viên nào để so sánh"
        )

    return {
        "status": STATUS_RUNNING,
        "query": query,
        "capabilities": capabilities,
        "warnings": warnings,
    }
