"""Node ``generate_embeddings``.

Với backend imintv3, embedding đã được tạo và lưu ở bước ``detect_faces``. Vậy node
này còn việc gì? Nó là **cổng kiểm soát chất lượng vector** trước khi truy vấn, và
nó là nơi duy nhất quyết định *những khuôn mặt nào* sẽ được đem đi tìm:

1. Xác định kết quả nào đang giữ embedding (lượt này, hay lượt trước).
2. Đọc embedding qua ``Principal`` - đây cũng chính là chốt chặn chống đọc kết quả
   của session khác: một ``result_id`` đoán được cũng không mở được dữ liệu.
3. Kiểm tra không gian embedding và số chiều khớp cấu hình hiện tại. Nếu ai đó đổi
   ``vision.embedding_space`` sau khi kết quả cũ đã lưu, câu truy vấn phải **lỗi**
   chứ không được trả về những con số trông hợp lệ.
4. Chọn index khuôn mặt theo yêu cầu ("khuôn mặt thứ hai"), báo lỗi rõ ràng nếu
   index không tồn tại.
5. Ghi lại những khuôn mặt bị loại và vì sao (quá nhỏ, thiếu vector).

Node này **không** đặt ma trận vector vào state - xem ghi chú ở ``FaceAgentState``.
Nó chỉ chốt danh sách ``query_face_indices``; node ``search_faces`` đọc lại vector.
"""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from app.agent.nodes.common import graph_node
from app.agent.state import FaceAgentContext, FaceAgentState
from app.errors import ValidationFailed


@graph_node("generate_embeddings")
def generate_embeddings(state: FaceAgentState, runtime: Runtime[FaceAgentContext]) -> dict[str, Any]:
    ctx = runtime.context

    source_id = state.get("detection_result_id") or state.get("target_result_id")
    if not source_id:
        raise ValidationFailed(
            "không có kết quả detection nào để lấy embedding (lượt này không có ảnh "
            "và thread không có kết quả cũ còn hiệu lực)",
            details={"missing": ["detection_result"]},
        )

    stored, space = ctx.deps.results.get_embeddings(source_id, principal=ctx.principal)

    warnings: list[str] = []
    if not stored:
        warnings.append(
            f"kết quả {source_id} không có embedding nào dùng được - không thể truy vấn. "
            "Thường là do ảnh không có khuôn mặt, hoặc mọi khuôn mặt đều quá nhỏ."
        )
        return {
            "embedding_source_result_id": source_id,
            "query_face_indices": [],
            "warnings": warnings,
        }

    requested = state.get("requested_face_indices")
    batch = ctx.deps.embedding.build_matrix_from_stored(
        stored,
        embedding_space=space,
        face_indices=requested,
    )

    for skipped in batch.skipped:
        warnings.append(
            f"face #{skipped['face_index']} bị loại khỏi truy vấn: {skipped['reason']}"
        )

    return {
        "embedding_source_result_id": source_id,
        "query_face_indices": list(batch.face_indices),
        "capabilities": {
            **(state.get("capabilities") or {}),
            "embedding_source_result_id": source_id,
            "embedding_space": batch.embedding_space,
            "available_face_indices": sorted(stored),
            "query_face_indices": list(batch.face_indices),
            "skipped_faces": list(batch.skipped),
        },
        "warnings": warnings,
    }
