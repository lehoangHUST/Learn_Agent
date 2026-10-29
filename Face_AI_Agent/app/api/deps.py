"""FastAPI dependency: principal, deps, runner, rate limit.

Đây là **lớp duy nhất** trong project biết tới FastAPI ngoài router và ``main.py``.
``app/agent``, ``app/services``, ``app/tools`` không import FastAPI, nên chúng chạy
được trong test, trong script CLI, hay sau này trong một worker queue.

Chính sách xác thực
-------------------
* Có ``FACE_AI_TOKENS`` -> bắt buộc bearer token, map sang ``Principal``.
* Không có và ``environment == "dev"`` -> cho phép principal ẩn danh, kèm WARNING ở
  log và một cảnh báo trong ``/health``. Đây là tiện lợi cho lúc phát triển.
* Không có và ``environment != "dev"`` -> **401**. Một hệ thống sinh trắc học không
  được tự mở cửa chỉ vì ai đó quên cấu hình token.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import Depends, Header, Request

from app.agent.runner import FaceAgentRunner
from app.dependencies import AgentDeps, get_agent, get_deps
from app.errors import Unauthenticated
from app.security import Principal, anonymous_principal

logger = logging.getLogger(__name__)
_anonymous_warned = False


def deps_dependency() -> AgentDeps:
    return get_deps()


DepsDep = Annotated[AgentDeps, Depends(deps_dependency)]


def principal_dependency(
    deps: DepsDep,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> Principal:
    global _anonymous_warned

    token: str | None = None
    if authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() != "bearer" or not value.strip():
            raise Unauthenticated("Authorization phải có dạng 'Bearer <token>'")
        token = value.strip()

    if deps.tokens.is_empty:
        if deps.settings.environment != "dev":
            raise Unauthenticated(
                "server chưa cấu hình FACE_AI_TOKENS nên không xác thực được; "
                "môi trường này không cho phép truy cập ẩn danh"
            )
        if not _anonymous_warned:
            logger.warning(
                "auth_disabled_dev_mode",
                extra={
                    "note": (
                        "FACE_AI_TOKENS rỗng và environment=dev -> dùng principal ẩn danh. "
                        "KHÔNG dùng cấu hình này ngoài máy phát triển."
                    )
                },
            )
            _anonymous_warned = True
        return anonymous_principal(deps.settings.vector_store.collection)

    return deps.tokens.resolve(token)


PrincipalDep = Annotated[Principal, Depends(principal_dependency)]


def rate_limit_dependency(request: Request, deps: DepsDep, principal: PrincipalDep) -> Principal:
    """Giới hạn theo principal, không theo IP.

    Theo IP thì một văn phòng dùng NAT chung sẽ chặn lẫn nhau, còn một kẻ lạm dụng
    đổi IP là thoát. Theo principal thì hạn mức đi theo đúng người gọi.
    """
    deps.rate_limiter.check(f"{principal.user_id}:{request.url.path}")
    return principal


RateLimitedPrincipalDep = Annotated[Principal, Depends(rate_limit_dependency)]


def runner_dependency(deps: DepsDep) -> FaceAgentRunner:
    return FaceAgentRunner(get_agent(), deps)


RunnerDep = Annotated[FaceAgentRunner, Depends(runner_dependency)]
