"""Hạ tầng dùng chung cho mọi node.

Decorator ``graph_node`` lo ba việc lặp lại ở mọi node, để thân node chỉ còn logic
nghiệp vụ:

1. **Log có cấu trúc**: ``node_name``, ``latency_ms``, ``status``, ``error_code``,
   kèm ``request_id``/``thread_id`` lấy từ state.
2. **Bắt lỗi thống nhất**: ``FaceAIError`` -> một ``ErrorEntry`` trong state, không
   làm sập graph. Exception lạ -> bọc thành lỗi nội bộ, cũng không làm sập graph.
   Vì sao không để exception bay lên: graph còn phải chạy ``handle_error`` và
   ``finalize_response`` để trả về một response đúng schema. Một exception thoát ra
   khỏi graph là một response 500 không có ``request_id`` - vô dụng cho người gọi và
   cho việc điều tra.
3. **Ghi trace**: mỗi node nối đúng một dòng vào field tích lũy ``trace`` và tên nó
   vào ``executed_steps``. Hai field này dùng reducer ``operator.add``, nên node chỉ
   trả về *phần thêm vào*, không bao giờ trả về cả danh sách cũ.

Tên node truyền vào decorator phải khớp tên đăng ký trong ``add_node``: đó cũng là
tên span hiện ra trong Langfuse.
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from typing import Any

from langgraph.runtime import Runtime

from app.agent.state import (
    STATUS_FAILED,
    ErrorEntry,
    FaceAgentContext,
)
from app.errors import FaceAIError

logger = logging.getLogger("app.agent.nodes")

NodeFn = Callable[[Any, Runtime[FaceAgentContext]], dict[str, Any] | None]


def error_entry(exc: FaceAIError, node: str) -> ErrorEntry:
    entry: ErrorEntry = {
        "code": exc.code,
        "message": exc.message,
        "retryable": exc.retryable,
        "node": node,
    }
    if exc.details:
        entry["details"] = dict(exc.details)
    return entry


def graph_node(name: str) -> Callable[[NodeFn], NodeFn]:
    def decorator(fn: NodeFn) -> NodeFn:
        @functools.wraps(fn)
        def wrapper(state: Any, runtime: Runtime[FaceAgentContext]) -> dict[str, Any]:
            started = time.perf_counter()
            base_fields = {
                "node_name": name,
                "request_id": _get(state, "request_id"),
                "thread_id": _get(state, "thread_id"),
            }
            try:
                update = fn(state, runtime) or {}
            except FaceAIError as exc:
                elapsed = _ms(started)
                logger.warning(
                    "node_failed",
                    extra={
                        **base_fields,
                        "status": "error",
                        "error_code": exc.code,
                        "retryable": exc.retryable,
                        "latency_ms": elapsed,
                    },
                )
                return {
                    "errors": [error_entry(exc, name)],
                    # Tín hiệu cho router: node VỪA chạy là node này, và nó lỗi.
                    "failed_node": name,
                    "status": STATUS_FAILED,
                    "trace": [f"{name}: error {exc.code} in {elapsed}ms"],
                    "executed_steps": [name],
                }
            except Exception as exc:
                elapsed = _ms(started)
                # exc_info=True: lỗi không lường trước thì phải có traceback trong log,
                # còn người gọi chỉ nhận mã lỗi chung chung - không rò chi tiết nội bộ.
                logger.exception(
                    "node_crashed",
                    extra={
                        **base_fields,
                        "status": "error",
                        "error_code": "internal_error",
                        "latency_ms": elapsed,
                    },
                )
                wrapped = FaceAIError(
                    f"lỗi không lường trước trong node {name}: {type(exc).__name__}",
                    details={"exception_type": type(exc).__name__},
                )
                return {
                    "errors": [error_entry(wrapped, name)],
                    "failed_node": name,
                    "status": STATUS_FAILED,
                    "trace": [f"{name}: crashed {type(exc).__name__} in {elapsed}ms"],
                    "executed_steps": [name],
                }

            elapsed = _ms(started)
            logger.info(
                "node_ok",
                extra={**base_fields, "status": "ok", "latency_ms": elapsed},
            )
            # Node có thể tự thêm trace riêng; phần của decorator được nối vào sau.
            node_trace = list(update.pop("trace", None) or [])
            update["trace"] = [*node_trace, f"{name}: ok in {elapsed}ms"]
            update["executed_steps"] = [*(update.pop("executed_steps", None) or []), name]
            # Xoá tín hiệu lỗi: node này chạy xong và không lỗi.
            #
            # Đây là mảnh ghép làm cho retry hoạt động. Lỗi cũ vẫn nằm trong
            # ``errors`` (đó là lịch sử, và người gọi cần thấy là đã có một lần lỗi),
            # nên router KHÔNG thể suy "node vừa chạy có lỗi không" từ ``errors[-1]``:
            # sau một lần retry thành công, ``errors[-1].node`` vẫn trỏ đúng vào node
            # vừa chạy thành công, và router sẽ lại đẩy về handle_error - vòng lặp
            # cho tới khi hết recursion_limit.
            update["failed_node"] = None
            return update

        return wrapper

    return decorator


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _get(state: Any, key: str) -> Any:
    if isinstance(state, dict):
        return state.get(key)
    return getattr(state, key, None)  # pragma: no cover - state luôn là dict ở đây
