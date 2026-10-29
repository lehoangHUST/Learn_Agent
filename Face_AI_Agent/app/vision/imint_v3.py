"""Adapter cho backend imintv3 ``POST /imintv3/extract_feature/``.

Hợp đồng thật của backend (đã đo trực tiếp, không suy đoán từ tài liệu)
----------------------------------------------------------------------
Request::

    {"base64_image": "<base64 không prefix>", "detect_face": 1, "is_checkresize": true}

Response HTTP 200::

    {
      "message": "success",            # <-- discriminator thành/bại
      "trace_message": "",
      "status_respond": 0,
      "result": {
        "features":         [[512 float], ...],   # 1 vector / mặt, ĐÃ L2-normalize
        "features_unalign": [[512 float], ...],   # KHÔNG gian khác, không trộn được
        "boxes":            [[x1, y1, x2, y2], ...],
        "landmarks":        [[x1,y1, ..., x5,y5], ...],
        "request":          {...echo lại cả base64 ảnh...}
      }
    }

Những điểm đã kiểm chứng và ảnh hưởng trực tiếp tới code dưới đây:

1. ``boxes`` chỉ có **4 số** - backend **không trả confidence**. Ta để
   ``confidence=None``; không bịa.
2. Ảnh không có mặt / mặt quá nhỏ (16px) -> **HTTP 200** nhưng ``message="Error"``
   và các list rỗng. ``status_respond`` vẫn là ``0`` trong cả hai trường hợp, nên
   **không được** dùng ``status_respond`` làm discriminator.
3. ``||features[i]|| == 1.0`` -> cosine similarity = inner product.
4. ``cos(features, features_unalign)`` trên cùng một mặt ~ **0.04**. Hai không gian
   khác nhau; trộn chúng trong một index là bug im lặng. Ta chốt theo
   ``vision.embedding_space`` và ghi không gian đó vào từng vector.
5. base64 rác -> **HTTP 500 với body text thuần** ``"Internal Server Error"``, không
   phải JSON. Phải bắt riêng, không ``.json()`` mù quáng.
6. ``result.request`` echo lại **toàn bộ base64 ảnh**. Bị loại bỏ ngay khi parse:
   không vào log, không vào state, không vào response.

Về detect vs embed: backend trả box + landmark + embedding trong **một** lời gọi,
nên ``analyze`` là primitive và ``detect``/``embed`` cùng gọi nó. Không có cache ở
đây (một embedding mặt người giữ trong RAM process là một bề mặt rủi ro không cần
thiết); thay vào đó graph được thiết kế để mỗi ảnh chỉ đi qua đúng một lời gọi -
xem ``app/agent/nodes/embed.py``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
import numpy as np

from app.config import VisionConfig
from app.errors import (
    VisionBackendError,
    VisionBackendRejectedImage,
    VisionBackendTimeout,
)
from app.vision.base import DetectedFace, FaceAnalysis, FaceBox
from app.vision.preprocessing import mark_small_faces, order_faces, parse_landmarks

logger = logging.getLogger(__name__)

_SUCCESS_MESSAGE = "success"
_FEATURE_KEY = {"aligned": "features", "unaligned": "features_unalign"}


class ImintV3FaceEngine:
    """Hiện thực ``BaseFaceEngine`` + ``BaseFaceDetector`` + ``BaseFaceEmbedder``."""

    name = "imint_v3"

    def __init__(self, config: VisionConfig, client: httpx.Client | None = None) -> None:
        self._config = config
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=config.base_url.rstrip("/"),
            timeout=httpx.Timeout(
                config.timeout_seconds,
                connect=config.connect_timeout_seconds,
            ),
        )

    # ------------------------------------------------------------------ #
    # BaseFaceEmbedder
    # ------------------------------------------------------------------ #
    @property
    def embedding_dim(self) -> int:
        return self._config.embedding_dim

    @property
    def embedding_space(self) -> str:
        return self._config.embedding_space

    # ------------------------------------------------------------------ #
    # BaseFaceEngine / Detector / Embedder - cùng một lời gọi HTTP
    # ------------------------------------------------------------------ #
    def detect(self, image_bytes: bytes) -> FaceAnalysis:
        return self.analyze(image_bytes)

    def embed(self, image_bytes: bytes) -> FaceAnalysis:
        return self.analyze(image_bytes)

    def analyze(self, image_bytes: bytes) -> FaceAnalysis:
        import base64

        payload = {
            "base64_image": base64.b64encode(image_bytes).decode("ascii"),
            "detect_face": self._config.detect_face,
            "is_checkresize": self._config.is_checkresize,
        }
        started = time.perf_counter()
        body = self._post_with_retry(payload)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return self._parse(body, latency_ms=latency_ms)

    def health(self) -> dict[str, Any]:
        """Kiểm tra backend còn sống bằng một GET nhẹ, không gửi ảnh nào."""
        try:
            resp = self._client.get("/openapi.json", timeout=httpx.Timeout(5.0, connect=3.0))
            return {
                "backend": self.name,
                "base_url": self._config.base_url,
                "reachable": resp.status_code == 200,
                "status_code": resp.status_code,
                "embedding_space": self.embedding_space,
                "embedding_dim": self.embedding_dim,
                "detector_confidence_available": False,
            }
        except httpx.HTTPError as exc:
            return {
                "backend": self.name,
                "base_url": self._config.base_url,
                "reachable": False,
                "error": type(exc).__name__,
                "embedding_space": self.embedding_space,
                "embedding_dim": self.embedding_dim,
                "detector_confidence_available": False,
            }

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # ------------------------------------------------------------------ #
    # HTTP
    # ------------------------------------------------------------------ #
    def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Gọi backend, retry có giới hạn và chỉ cho những lỗi retry được.

        Không retry lỗi 4xx: backend đã nói payload sai, gọi lại cùng payload thì
        vẫn sai. Có retry timeout và 5xx vì ảnh đã được validate phía ta trước khi
        gửi, nên 5xx ở đây là vấn đề phía server chứ không phải payload.
        """
        path = self._config.extract_feature_path
        attempts = max(1, self._config.max_retries + 1)
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                resp = self._client.post(path, json=payload)
            except httpx.TimeoutException:
                last_error = VisionBackendTimeout(
                    f"backend {self.name} timeout sau {self._config.timeout_seconds}s",
                    details={"attempt": attempt},
                )
            except httpx.HTTPError as exc:
                last_error = VisionBackendError(
                    f"không gọi được backend {self.name}: {type(exc).__name__}",
                    details={"attempt": attempt},
                )
            else:
                if resp.status_code == 200:
                    try:
                        body = resp.json()
                    except ValueError as exc:
                        raise VisionBackendError(
                            "backend trả HTTP 200 nhưng body không phải JSON",
                            details={"body_prefix": resp.text[:200]},
                        ) from exc
                    if not isinstance(body, dict):
                        raise VisionBackendError(
                            "backend trả JSON không phải object",
                            details={"type": type(body).__name__},
                        )
                    return body

                if 400 <= resp.status_code < 500:
                    # Backend từ chối payload -> lỗi đầu vào, không retry.
                    raise VisionBackendRejectedImage(
                        f"backend từ chối ảnh (HTTP {resp.status_code})",
                        details={
                            "status_code": resp.status_code,
                            "body_prefix": resp.text[:200],
                        },
                    )

                last_error = VisionBackendError(
                    f"backend lỗi HTTP {resp.status_code}",
                    details={
                        "status_code": resp.status_code,
                        "body_prefix": resp.text[:200],
                        "attempt": attempt,
                    },
                )

            if attempt < attempts:
                delay = self._config.retry_backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "vision_backend_retry",
                    extra={
                        "backend": self.name,
                        "attempt": attempt,
                        "max_attempts": attempts,
                        "sleep_s": delay,
                        "error_code": getattr(last_error, "code", None),
                    },
                )
                time.sleep(delay)

        assert last_error is not None  # vòng lặp luôn chạy >= 1 lần
        raise last_error

    # ------------------------------------------------------------------ #
    # Parsing
    # ------------------------------------------------------------------ #
    def _parse(self, body: dict[str, Any], *, latency_ms: float) -> FaceAnalysis:
        message = str(body.get("message", ""))
        trace = str(body.get("trace_message") or "")
        result = body.get("result")

        if not isinstance(result, dict):
            raise VisionBackendError(
                "response thiếu object 'result'",
                details={"message": message, "trace_message": trace[:200]},
            )

        # result.request echo lại toàn bộ base64 ảnh -> bỏ ngay, không đi đâu nữa.
        result.pop("request", None)

        boxes = result.get("boxes") or []
        landmarks = result.get("landmarks") or []
        feature_key = _FEATURE_KEY[self._config.embedding_space]
        features = result.get(feature_key) or []

        if not isinstance(boxes, list):
            raise VisionBackendError("'boxes' không phải list", details={"type": type(boxes).__name__})

        warnings: list[str] = []

        # Không có mặt nào: backend báo message="Error" nhưng đây là kết quả hợp lệ.
        if len(boxes) == 0:
            if message != _SUCCESS_MESSAGE:
                warnings.append(
                    f"backend báo message={message!r} khi không phát hiện khuôn mặt nào "
                    "(hành vi đã biết của imintv3, không phải lỗi hệ thống)"
                )
            return FaceAnalysis(
                faces=[],
                backend=self.name,
                backend_message=message,
                latency_ms=latency_ms,
                warnings=warnings,
            )

        if message != _SUCCESS_MESSAGE:
            warnings.append(f"backend trả message={message!r} nhưng vẫn có {len(boxes)} box")

        if len(features) != len(boxes):
            warnings.append(
                f"số vector ({len(features)}) không khớp số box ({len(boxes)}); "
                "các mặt thiếu vector sẽ không dùng được cho search"
            )

        faces: list[DetectedFace] = []
        for idx, raw_box in enumerate(boxes):
            try:
                box = FaceBox.from_sequence(raw_box)
            except (TypeError, ValueError) as exc:
                warnings.append(f"bỏ qua box #{idx} không hợp lệ: {exc}")
                continue

            embedding = None
            if idx < len(features):
                embedding = self._to_vector(features[idx], idx, warnings)

            faces.append(
                DetectedFace(
                    face_index=idx,  # gán lại sau order_faces
                    box=box,
                    landmarks=parse_landmarks(landmarks[idx] if idx < len(landmarks) else None),
                    confidence=None,  # backend không trả confidence - xem docstring module
                    confidence_source="unavailable_from_backend",
                    embedding=embedding,
                    embedding_space=self.embedding_space if embedding is not None else None,
                )
            )

        faces = order_faces(faces)
        warnings.extend(mark_small_faces(faces, self._config.min_face_size_px))

        for face in faces:
            if face.embedding is None and face.rejected_reason is None:
                face.rejected_reason = "missing_embedding_from_backend"

        return FaceAnalysis(
            faces=faces,
            backend=self.name,
            backend_message=message,
            latency_ms=latency_ms,
            warnings=warnings,
        )

    def _to_vector(self, raw: Any, idx: int, warnings: list[str]) -> np.ndarray | None:
        """Chuyển list float -> vector float32 đã kiểm tra dim và norm."""
        try:
            vec = np.asarray(raw, dtype=np.float32)
        except (TypeError, ValueError):
            warnings.append(f"vector của mặt #{idx} không parse được")
            return None

        if vec.ndim != 1:
            warnings.append(f"vector của mặt #{idx} có shape {vec.shape}, cần 1-D")
            return None

        if vec.shape[0] != self._config.embedding_dim:
            # Không im lặng chấp nhận: index đã cố định dim, nhận sai dim là bug cấu hình.
            raise VisionBackendError(
                f"backend trả vector {vec.shape[0]}-D nhưng vision.embedding_dim = "
                f"{self._config.embedding_dim}. Sửa config và build lại index.",
                details={"got_dim": int(vec.shape[0]), "expected_dim": self._config.embedding_dim},
            )

        if not np.all(np.isfinite(vec)):
            warnings.append(f"vector của mặt #{idx} chứa NaN/Inf, bị loại")
            return None

        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            warnings.append(f"vector của mặt #{idx} là vector 0, bị loại")
            return None

        if self._config.enforce_l2_normalize and abs(norm - 1.0) > 1e-3:
            # Backend thật đã normalize; nhánh này là bảo hiểm khi đổi model.
            warnings.append(f"vector của mặt #{idx} có norm {norm:.4f}, đã normalize lại")
            vec = vec / norm

        return vec
