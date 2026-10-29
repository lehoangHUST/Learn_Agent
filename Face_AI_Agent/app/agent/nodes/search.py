"""Node ``search_faces``.

Đọc lại vector của những khuôn mặt mà ``generate_embeddings`` đã chốt, rồi truy vấn
kho vector **một lần cho tất cả** (xem ``BaseFaceSearchRepository.search``). Không
fan-out song song cho từng khuôn mặt: FAISS tìm theo batch nhanh hơn n lần tìm lẻ,
và một node tuần tự thì dễ đọc hơn một super-step có n nhánh làm cùng một việc.

Phân quyền ở ba tầng, cố ý trùng nhau:

1. ``results.get_embeddings`` - chỉ chủ sở hữu đọc được embedding của kết quả.
2. ``principal.require_collection`` trong repository - chỉ truy vấn collection được
   phép.
3. Lọc từng record theo ``visibility``/``owner_id`` khi ghép kết quả.

Trùng lặp ở đây là có chủ ý: mỗi tầng chặn một kiểu sai khác nhau, và không tầng nào
phải tin tầng trước đã làm đúng.

Node này **không** kết luận gì. Việc gán ``match``/``ambiguous``/``no_match`` nằm
trong ``FaceSearchService``, và việc diễn giải nằm ở ``analyze_results``.
"""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from app.agent.nodes.common import graph_node
from app.agent.state import FaceAgentContext, FaceAgentState
from app.errors import ValidationFailed
from app.repositories.result_store import KIND_SEARCH


@graph_node("search_faces")
def search_faces(state: FaceAgentState, runtime: Runtime[FaceAgentContext]) -> dict[str, Any]:
    ctx = runtime.context

    source_id = state.get("embedding_source_result_id")
    if not source_id:
        raise ValidationFailed(
            "search_faces được gọi mà chưa có nguồn embedding",
            details={"missing": ["embedding_source_result_id"]},
        )

    query_indices = state.get("query_face_indices")
    stored, space = ctx.deps.results.get_embeddings(source_id, principal=ctx.principal)
    batch = ctx.deps.embedding.build_matrix_from_stored(
        stored,
        embedding_space=space,
        face_indices=query_indices if query_indices is not None else None,
    )

    outcome = ctx.deps.search.search(
        batch,
        principal=ctx.principal,
        collection=ctx.collection,
        top_k=ctx.deps.settings.matching.top_k,
    )
    payload = outcome.to_public_dict()
    payload["kind"] = KIND_SEARCH
    payload["source_detection_result_id"] = source_id

    result_id = ctx.deps.results.save(
        kind=KIND_SEARCH,
        owner_id=ctx.principal.user_id,
        payload=payload,
        thread_id=state.get("thread_id"),
        request_id=state.get("request_id"),
    )

    return {
        "search_result_id": result_id,
        "search_summary": payload,
        "warnings": list(outcome.warnings),
    }
