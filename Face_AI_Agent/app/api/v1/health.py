"""Endpoint ``/api/v1/health``.

Không trả "ok" một cách dễ dãi. Health ở đây nói ba chuyện mà người vận hành cần:

1. Backend vision có gọi tới được không (``reachable``).
2. Index có dữ liệu không - một hệ thống search với index rỗng vẫn "up" nhưng không
   trả lời được gì, nên đó là ``degraded``, không phải ``ok``.
3. **Ngưỡng so khớp đã hiệu chuẩn chưa.** Đây là thông tin vận hành quan trọng nhất
   của một hệ thống nhận dạng và nó được đưa lên ngay cấp cao nhất của response.
"""

from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.api.deps import DepsDep
from app.schemas.common import ComponentHealth, HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Trạng thái ứng dụng")
def health(deps: DepsDep) -> HealthResponse:
    raw = deps.health()
    settings = deps.settings

    vision = raw["vision"]
    vector = raw["vector_store"]
    collection = settings.vector_store.collection
    index_count = (vector.get("collections") or {}).get(collection, 0)

    components = [
        ComponentHealth(name="vision_backend", ok=bool(vision.get("reachable")), detail=vision),
        ComponentHealth(name="vector_store", ok=True, detail=vector),
        ComponentHealth(name="result_store", ok=True, detail=raw["result_store"]),
        ComponentHealth(name="image_store", ok=True, detail=raw["image_store"]),
        ComponentHealth(
            name="llm",
            ok=True,  # LLM tắt không phải lỗi: hệ thống chạy đủ chức năng ở chế độ template
            detail=raw["llm"],
        ),
    ]

    warnings: list[str] = []
    if not vision.get("reachable"):
        warnings.append(
            f"không gọi được backend vision tại {settings.vision.base_url} - "
            "mọi yêu cầu phát hiện/tìm kiếm sẽ lỗi"
        )
    if index_count == 0:
        warnings.append(
            f"collection {collection!r} chưa có khuôn mặt nào: tìm kiếm sẽ luôn trả "
            "no_candidates. Nạp dữ liệu bằng scripts/register_face.py"
        )
    if not settings.matching.calibrated:
        warnings.append(
            "match_threshold chưa được hiệu chuẩn: kết quả 'match' chỉ nên coi là ứng "
            "viên cần người xác nhận (scripts/calibrate_threshold.py)"
        )
    if deps.tokens.is_empty and settings.environment == "dev":
        warnings.append("xác thực đang TẮT (FACE_AI_TOKENS rỗng, environment=dev)")

    status = "ok" if all(c.ok for c in components) and not warnings else "degraded"

    return HealthResponse(
        status=status,
        app_name=settings.app_name,
        version=__version__,
        environment=settings.environment,
        components=components,
        matching_threshold_calibrated=settings.matching.calibrated,
        warnings=warnings,
    )
