"""Node ``detect_faces``.

Gọi ``FaceDetectionService`` cho từng ảnh, rồi **lưu kết quả có cấu trúc** vào result
store: payload công khai (box, landmark, kích thước) đi một đường, embedding đi một
đường khác.

Vì sao node detect lại lưu cả embedding
---------------------------------------
Backend imintv3 trả box + landmark + embedding trong *một* lời gọi. Nếu không lưu
embedding ngay, node ``generate_embeddings`` sẽ phải gọi backend lần thứ hai cho
cùng một ảnh - gấp đôi thời gian và gấp đôi tải GPU, chỉ để lấy lại thứ vừa có
trong tay. Lưu ngay cũng là thứ làm cho luồng hỏi tiếp ở lượt sau ("tìm khuôn mặt
thứ hai") chạy được mà không cần người dùng gửi lại ảnh.

Đánh đổi: embedding sinh trắc học nằm trên đĩa trong ``storage.result_ttl_seconds``.
Đó là lý do TTL tồn tại, là lý do bảng embedding tách khỏi payload, và là lý do mọi
lần đọc đều đi qua ``Principal``.

Đánh số khuôn mặt
-----------------
``global_face_index`` chạy suốt các ảnh trong một request; ``face_index`` là chỉ số
trong từng ảnh. Với một ảnh (trường hợp thường gặp) hai số này trùng nhau. Mọi tham
chiếu về sau - ``requested_face_indices``, khoá embedding, kết quả search - đều dùng
``global_face_index``, để "khuôn mặt thứ hai" chỉ có đúng một nghĩa.
"""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from app.agent.nodes.common import graph_node
from app.agent.state import FaceAgentContext, FaceAgentState
from app.errors import ValidationFailed
from app.repositories.result_store import KIND_DETECTION


@graph_node("detect_faces")
def detect_faces(state: FaceAgentState, runtime: Runtime[FaceAgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    settings = ctx.deps.settings
    image_refs = list(state.get("image_refs") or [])

    if not image_refs:
        raise ValidationFailed("detect_faces được gọi nhưng request không có ảnh nào")

    images_payload: list[dict[str, Any]] = []
    faces_flat: list[dict[str, Any]] = []
    embeddings: dict[int, Any] = {}
    warnings: list[str] = []
    global_index = 0

    for image_index, ref in enumerate(image_refs):
        image = ctx.images.get(ref["ref"])
        if image is None:  # pragma: no cover - validate_input đã chặn
            raise ValidationFailed(
                "image_ref không giải được", details={"image_index": image_index}
            )

        outcome = ctx.deps.detection.detect(image)
        # Với imintv3 thì embedding đã có sẵn -> hàm này chỉ kiểm tra, không gọi thêm HTTP.
        ctx.deps.embedding.ensure_embeddings(outcome.analysis, image.raw)

        per_image: dict[str, Any] = {
            "image_index": image_index,
            "image_ref": ref["ref"],
            "image": {
                "sha256": image.sha256,
                "mime": image.mime,
                "width": image.width,
                "height": image.height,
                "size_bytes": image.size_bytes,
            },
            "face_count": outcome.analysis.face_count,
            "usable_face_count": len(outcome.analysis.usable_faces),
            "backend": outcome.analysis.backend,
            "backend_message": outcome.analysis.backend_message,
            "latency_ms": outcome.analysis.latency_ms,
            "faces": [],
            "warnings": list(outcome.analysis.warnings),
        }

        for face in outcome.analysis.faces:
            public = face.to_public_dict(image.width, image.height)
            public["global_face_index"] = global_index
            public["image_index"] = image_index
            per_image["faces"].append(public)
            faces_flat.append(public)

            if face.is_usable and face.embedding is not None:
                embeddings[global_index] = face.embedding
            global_index += 1

        warnings.extend(f"ảnh #{image_index}: {w}" for w in outcome.analysis.warnings)
        images_payload.append(per_image)

        # Mặc định KHÔNG lưu ảnh truy vấn. Chỉ lưu khi được bật tường minh.
        if settings.storage.persist_query_images:
            stored = ctx.deps.images.save(
                image.raw, mime=image.mime, owner_id=ctx.principal.user_id
            )
            per_image["stored_image_ref"] = stored.image_ref

    total_faces = len(faces_flat)
    usable = sum(1 for f in faces_flat if f.get("usable_for_search"))

    payload: dict[str, Any] = {
        "kind": KIND_DETECTION,
        "image_count": len(image_refs),
        "total_face_count": total_faces,
        "usable_face_count": usable,
        # Nói thẳng ra trong payload: nguồn dữ liệu không có confidence. Node answer
        # và cả LLM đều đọc field này thay vì phải suy luận từ việc confidence là null.
        "detector_confidence_available": any(f.get("confidence") is not None for f in faces_flat),
        "confidence_note": (
            "backend không trả confidence của detector; chỉ có kích thước bounding box, "
            "và kích thước KHÔNG phải confidence"
        ),
        "embedding_space": ctx.deps.embedding.embedding_space,
        "embedding_dim": ctx.deps.embedding.embedding_dim,
        "images": images_payload,
        "faces": faces_flat,
        "warnings": warnings,
    }

    if total_faces == 0:
        warnings.append(
            "không phát hiện được khuôn mặt nào. Nguyên nhân thường gặp: ảnh không có "
            "mặt người, mặt quá nhỏ so với ảnh, hoặc mặt bị che/mờ."
        )
        payload["warnings"] = warnings

    result_id = ctx.deps.results.save(
        kind=KIND_DETECTION,
        owner_id=ctx.principal.user_id,
        payload=payload,
        thread_id=state.get("thread_id"),
        request_id=state.get("request_id"),
        embeddings=embeddings or None,
        embedding_space=ctx.deps.embedding.embedding_space if embeddings else None,
    )

    return {
        "detection_result_id": result_id,
        "detection_summary": payload,
        # Kết quả vừa tạo trở thành mục tiêu mặc định cho các bước sau trong lượt này.
        "target_result_id": result_id,
        "warnings": warnings,
    }
