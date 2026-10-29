"""Ứng dụng FastAPI.

Ba việc chính ở đây:

1. **Lifespan**: dựng dependency một lần khi khởi động (compile graph, nạp index,
   mở kết nối checkpoint), dọn dữ liệu hết hạn, và đóng tài nguyên khi tắt.
   Compile graph lúc khởi động là có chủ ý: ``compile()`` kiểm tra topology, nên
   graph sai cấu trúc làm app không khởi động được thay vì lỗi giữa một request.

2. **Exception handler**: chuyển ``FaceAIError`` thành đúng HTTP status và một body
   ``ErrorResponse`` thống nhất. Đây là nơi duy nhất trong hệ thống biết tới HTTP
   status - tầng dưới chỉ biết mã lỗi và ``retryable``.

3. **Middleware request_id**: mỗi request có ``request_id``, trả trong header
   ``X-Request-Id`` và có trong mọi dòng log của request đó.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app import __version__
from app.api.v1 import chat, faces, health, results
from app.config import load_settings
from app.dependencies import get_agent, get_deps, reset_singletons
from app.errors import FaceAIError
from app.logging_utils import configure_logging, new_request_id

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = load_settings()
    configure_logging(
        settings.observability.log_level, json_logs=settings.observability.json_logs
    )

    deps = get_deps()
    get_agent()

    purged_results = deps.results.purge_expired()
    purged_images = deps.images.purge_expired()

    logger.info(
        "app_started",
        extra={
            "version": __version__,
            "environment": settings.environment,
            "vision_backend": settings.vision.backend,
            "embedding_space": settings.vision.embedding_space,
            "threshold_calibrated": settings.matching.calibrated,
            "llm_enabled": settings.llm.enabled,
            "purged_results": purged_results,
            "purged_images": purged_images,
        },
    )
    try:
        yield
    finally:
        reset_singletons()
        logger.info("app_stopped")


app = FastAPI(
    title="Face AI Agent",
    version=__version__,
    description=(
        "Face detection, face search (1:N) và trả lời câu hỏi về kết quả, điều phối "
        "bằng LangGraph.\n\n"
        "**Những điều cần biết trước khi dùng:**\n\n"
        "- Backend detector **không trả confidence score**. Field `confidence` luôn "
        "`null`; kích thước bounding box không phải confidence.\n"
        "- `similarity_score` là **cosine similarity** trong [-1, 1], không phải xác "
        "suất cùng danh tính.\n"
        "- `match_status` có 5 giá trị; `ambiguous` và `needs_review` nghĩa là "
        "**không kết luận được**, phải để người xem lại.\n"
        "- Ngưỡng so khớp mặc định **chưa được hiệu chuẩn** - xem `/health`.\n"
        "- Response không bao giờ chứa embedding hay ảnh gốc."
    ),
    lifespan=lifespan,
)

app.include_router(faces.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(results.router, prefix="/api/v1")
app.include_router(health.router, prefix="/api/v1")


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    request_id = request.headers.get("X-Request-Id") or new_request_id()
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or new_request_id()


@app.exception_handler(FaceAIError)
async def face_ai_error_handler(request: Request, exc: FaceAIError) -> JSONResponse:
    """Lỗi có kiểm soát -> status và body thống nhất.

    ``NoFaceDetected``/``EmptyIndex`` có ``http_status = 200`` vì chúng là *kết quả*,
    không phải lỗi; nếu chúng tới được đây thì nghĩa là ai đó raise sai chỗ, nên vẫn
    log ở mức warning.
    """
    request_id = _request_id(request)
    logger.warning(
        "api_error",
        extra={
            "request_id": request_id,
            "error_code": exc.code,
            "status": exc.http_status,
            "path": request.url.path,
            "retryable": exc.retryable,
        },
    )
    return JSONResponse(
        status_code=exc.http_status,
        content={"request_id": request_id, "error": exc.to_dict()},
        headers={"X-Request-Id": request_id},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Lỗi schema của FastAPI -> cùng hình dạng ErrorResponse.

    Chi tiết của pydantic được giữ lại trong ``details`` nhưng phần ``input`` bị bỏ:
    với endpoint nhận base64 ảnh, ``input`` sẽ là cả tấm ảnh trong body lỗi và trong
    log - đúng thứ không được xảy ra.
    """
    request_id = _request_id(request)
    safe_errors = [
        {"loc": err.get("loc"), "msg": err.get("msg"), "type": err.get("type")}
        for err in exc.errors()
    ]
    logger.warning(
        "api_validation_error",
        extra={"request_id": request_id, "error_code": "schema_validation", "path": request.url.path},
    )
    return JSONResponse(
        status_code=422,
        content={
            "request_id": request_id,
            "error": {
                "code": "schema_validation",
                "message": "request không đúng schema",
                "retryable": False,
                "details": {"errors": safe_errors},
            },
        },
        headers={"X-Request-Id": request_id},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Chốt chặn cuối: log đầy đủ traceback, trả ra ngoài thông tin tối thiểu."""
    request_id = _request_id(request)
    logger.exception(
        "api_unhandled_error",
        extra={
            "request_id": request_id,
            "error_code": "internal_error",
            "path": request.url.path,
        },
    )
    return JSONResponse(
        status_code=500,
        content={
            "request_id": request_id,
            "error": {
                "code": "internal_error",
                "message": "lỗi nội bộ; dùng request_id để tra log",
                "retryable": True,
                "details": {},
            },
        },
        headers={"X-Request-Id": request_id},
    )
