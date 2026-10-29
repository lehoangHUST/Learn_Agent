"""Structured logging + redaction.

Hai việc:

1. Log ra JSON với các field cố định: ``request_id``, ``thread_id``, ``node_name``,
   ``latency_ms``, ``status``, ``error_code``. Nhờ vậy log query được.
2. **Chặn dữ liệu sinh trắc học rò ra log.** Đây không phải tuỳ chọn: ảnh base64,
   embedding và API key không bao giờ được xuất hiện trong log. ``_scrub`` cắt mọi
   key nằm trong ``_SENSITIVE_KEYS`` và mọi chuỗi dài nghi là base64.

Backend imintv3 echo lại *toàn bộ* base64 ảnh trong ``result.request`` - nếu log
nguyên response thì toàn bộ ảnh mặt người nằm trong log file. ``_scrub`` tồn tại
chủ yếu vì chuyện đó.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_SENSITIVE_KEYS = {
    "base64_image",
    "base64",
    "image_base64",
    "images_base64",
    "embedding",
    "embeddings",
    "features",
    "features_unalign",
    "vector",
    "vectors",
    "api_key",
    "openai_api_key",
    "authorization",
    "password",
    "token",
    "secret",
}

# Chuỗi dài hơn mức này bị coi là payload và cắt bớt, dù key có tên gì.
_MAX_STR_LEN = 256


def _scrub(value: Any, _depth: int = 0) -> Any:
    """Trả về bản copy an toàn để log của ``value``."""
    if _depth > 6:
        return "<max-depth>"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, val in value.items():
            if str(key).lower() in _SENSITIVE_KEYS:
                out[key] = _summarize_redacted(val)
            else:
                out[key] = _scrub(val, _depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        if len(value) > 20:
            return [_scrub(v, _depth + 1) for v in value[:20]] + [f"<+{len(value) - 20} more>"]
        return [_scrub(v, _depth + 1) for v in value]
    if isinstance(value, str) and len(value) > _MAX_STR_LEN:
        return f"<str len={len(value)} truncated>"
    return value


def _summarize_redacted(value: Any) -> str:
    """Không log nội dung, nhưng vẫn log *hình dạng* để debug được."""
    if isinstance(value, str):
        return f"<redacted str len={len(value)}>"
    if isinstance(value, (list, tuple)):
        return f"<redacted list len={len(value)}>"
    if value is None:
        return "<redacted none>"
    return f"<redacted {type(value).__name__}>"


_RESERVED = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, val in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = _scrub(val)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class PlainFormatter(logging.Formatter):
    """Dùng khi dev muốn đọc bằng mắt. Vẫn scrub."""

    def format(self, record: logging.LogRecord) -> str:
        extras = {
            k: _scrub(v)
            for k, v in record.__dict__.items()
            if k not in _RESERVED and not k.startswith("_")
        }
        tail = " ".join(f"{k}={v}" for k, v in extras.items())
        base = f"{record.levelname:<7} {record.name} | {record.getMessage()}"
        return f"{base} | {tail}" if tail else base


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_logs else PlainFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())
    # httpx log mọi request ở INFO, kèm URL - ồn và không cần thiết.
    logging.getLogger("httpx").setLevel("WARNING")
    logging.getLogger("httpcore").setLevel("WARNING")


def new_request_id() -> str:
    return "req_" + uuid.uuid4().hex[:16]


def new_thread_id() -> str:
    return "thr_" + uuid.uuid4().hex[:16]


def new_result_id(kind: str) -> str:
    return f"{kind}_" + uuid.uuid4().hex[:16]


@contextmanager
def timed(logger: logging.Logger, event: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Log một event kèm ``latency_ms`` và ``status``, kể cả khi raise.

    ``extra`` trả ra cho caller ghi thêm field vào bản log cuối.
    """
    start = time.perf_counter()
    extra: dict[str, Any] = {}
    try:
        yield extra
    except Exception as exc:
        logger.warning(
            event,
            extra={
                **fields,
                **extra,
                "status": "error",
                "error_code": getattr(exc, "code", exc.__class__.__name__),
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            },
        )
        raise
    else:
        logger.info(
            event,
            extra={
                **fields,
                **extra,
                "status": "ok",
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            },
        )
