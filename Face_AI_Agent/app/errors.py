"""Error taxonomy cho toàn hệ thống.

Mục tiêu: mỗi lỗi mang 3 thông tin mà cả graph, API và logging đều cần:

* ``code``      - mã ổn định để client/log phân loại (không bao giờ đổi theo message).
* ``retryable`` - lỗi này gọi lại có cơ hội thành công hay không. Node ``handle_error``
                  dùng đúng field này để quyết định retry, không suy đoán từ message.
* ``http_status``- status code mà API layer trả ra.

Nguyên tắc: business/vision/agent chỉ raise các exception ở đây; chỉ tầng API mới
biết tới HTTP. Nhờ vậy ``app/agent`` và ``app/services`` không phụ thuộc FastAPI.
"""

from __future__ import annotations

from typing import Any


class FaceAIError(Exception):
    """Base cho mọi lỗi có kiểm soát của hệ thống."""

    code: str = "internal_error"
    retryable: bool = False
    http_status: int = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
        }


# --------------------------------------------------------------------------- #
# 4xx - lỗi do đầu vào. Retry không bao giờ giúp được.
# --------------------------------------------------------------------------- #
class ValidationFailed(FaceAIError):
    code = "invalid_input"
    retryable = False
    http_status = 422


class ImageTooLarge(ValidationFailed):
    code = "image_too_large"
    http_status = 413


class UnsupportedImageType(ValidationFailed):
    code = "unsupported_image_type"


class ImageDecodeFailed(ValidationFailed):
    code = "image_decode_failed"


class AccessDenied(FaceAIError):
    code = "access_denied"
    retryable = False
    http_status = 403


class Unauthenticated(FaceAIError):
    code = "unauthenticated"
    retryable = False
    http_status = 401


class ResultNotFound(FaceAIError):
    code = "result_not_found"
    retryable = False
    http_status = 404


class RateLimited(FaceAIError):
    code = "rate_limited"
    retryable = True
    http_status = 429


class UnsupportedRequest(FaceAIError):
    code = "unsupported_request"
    retryable = False
    http_status = 400


# --------------------------------------------------------------------------- #
# 5xx / phụ thuộc ngoài - một số retry được.
# --------------------------------------------------------------------------- #
class VisionBackendError(FaceAIError):
    """Backend face (imintv3) trả lỗi không hiểu được."""

    code = "vision_backend_error"
    retryable = True
    http_status = 502


class VisionBackendTimeout(VisionBackendError):
    code = "vision_backend_timeout"
    retryable = True
    http_status = 504


class VisionBackendRejectedImage(VisionBackendError):
    """Backend trả 5xx vì payload ảnh không giải mã được phía nó.

    Gọi lại cùng payload sẽ lỗi y hệt ⇒ KHÔNG retryable, dù status là 5xx.
    """

    code = "vision_backend_rejected_image"
    retryable = False
    http_status = 422


class NoFaceDetected(FaceAIError):
    """Không phải lỗi hệ thống - là một kết quả hợp lệ, nhưng cần đi nhánh riêng."""

    code = "no_face_detected"
    retryable = False
    http_status = 200


class VectorStoreError(FaceAIError):
    code = "vector_store_error"
    retryable = True
    http_status = 503


class EmptyIndex(FaceAIError):
    code = "empty_index"
    retryable = False
    http_status = 200


class LLMError(FaceAIError):
    code = "llm_error"
    retryable = True
    http_status = 502


class LLMNotConfigured(FaceAIError):
    code = "llm_not_configured"
    retryable = False
    http_status = 501
