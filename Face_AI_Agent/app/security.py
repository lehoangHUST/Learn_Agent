"""Principal, phân quyền và rate limit.

Hệ thống xử lý dữ liệu sinh trắc học, nên quyền truy cập không phải thứ bọc thêm
vào sau: **mọi** truy vấn index và mọi lần đọc kết quả đã lưu đều phải đi qua một
``Principal``. Repository và result store nhận principal như tham số bắt buộc, chứ
không nhận ``user_id`` dạng string tuỳ ý - như vậy không thể "quên" kiểm tra.

Phần xác thực ở đây cố tình đơn giản (bearer token tĩnh map sang principal) và được
đánh dấu rõ là DEMO. Nó đủ để các bất biến về phân quyền được hiện thực và test
thật; đổi sang OIDC/JWT chỉ cần thay ``resolve_principal``.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

from app.errors import AccessDenied, RateLimited, Unauthenticated

ROLE_ADMIN = "admin"
ROLE_SEARCH = "face:search"
ROLE_REGISTER = "face:register"
ROLE_DELETE = "face:delete"
ROLE_READ_METADATA = "face:read_metadata"


@dataclass(frozen=True, slots=True)
class Principal:
    """Chủ thể đã xác thực của một request."""

    user_id: str
    roles: frozenset[str] = field(default_factory=frozenset)
    # Collection (tập dữ liệu khuôn mặt) mà principal được phép đọc.
    allowed_collections: frozenset[str] = field(default_factory=frozenset)
    tenant_id: str | None = None

    @property
    def is_admin(self) -> bool:
        return ROLE_ADMIN in self.roles

    def has_role(self, role: str) -> bool:
        return self.is_admin or role in self.roles

    def require_role(self, role: str) -> None:
        if not self.has_role(role):
            raise AccessDenied(
                f"principal {self.user_id} thiếu quyền {role}",
                details={"required_role": role},
            )

    def can_read_collection(self, collection: str) -> bool:
        return self.is_admin or collection in self.allowed_collections

    def require_collection(self, collection: str) -> None:
        if not self.can_read_collection(collection):
            raise AccessDenied(
                f"principal {self.user_id} không được phép truy cập collection {collection!r}",
                details={"collection": collection},
            )

    def require_owner(self, owner_id: str, *, what: str = "resource") -> None:
        """Chỉ chủ sở hữu (hoặc admin) được đọc.

        Đây là chốt chặn cho trường hợp một người cố đọc kết quả của session khác:
        lỗi là 403, và quan trọng hơn - thông báo không tiết lộ gì về nội dung
        resource đó.
        """
        if self.is_admin or owner_id == self.user_id:
            return
        raise AccessDenied(
            f"principal {self.user_id} không phải chủ sở hữu của {what} này",
            details={"what": what},
        )

    def log_fields(self) -> dict[str, object]:
        return {"user_id": self.user_id, "tenant_id": self.tenant_id}


# Principal dùng khi chạy script CLI / job nội bộ.
SYSTEM_PRINCIPAL = Principal(
    user_id="system",
    roles=frozenset({ROLE_ADMIN}),
    allowed_collections=frozenset(),
)


def anonymous_principal(collection: str) -> Principal:
    """Principal cho chế độ dev không bật auth. KHÔNG dùng ở prod."""
    return Principal(
        user_id="dev-anonymous",
        roles=frozenset({ROLE_SEARCH, ROLE_REGISTER, ROLE_READ_METADATA}),
        allowed_collections=frozenset({collection}),
    )


# --------------------------------------------------------------------------- #
# Xác thực - DEMO
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class TokenPrincipalStore:
    """Map bearer token -> principal.

    DEMO: token nạp từ env ``FACE_AI_TOKENS``. Cú pháp::

        <token>|<user_id>|<role,role,...>|<collection,collection,...>

    Nhiều principal phân tách bằng ``;``. Ví dụ::

        FACE_AI_TOKENS="t-alice|alice|face:search,face:register|default;t-adm|adm|admin|"

    Dấu phân tách là ``|`` và ``,`` chứ **không** phải ``:``, vì tên quyền đã chứa
    ``:`` (``face:search``) - dùng ``:`` làm phân tách sẽ cắt nát tên quyền và
    principal nhận quyền sai mà không có lỗi nào được báo.

    Production: thay bằng xác thực JWT/OIDC và nguồn quyền thật (IdP, DB). Chỉ cần
    thay ``resolve``.
    """

    tokens: dict[str, Principal]

    @classmethod
    def from_env(cls, raw: str | None = None) -> TokenPrincipalStore:
        raw = raw if raw is not None else os.environ.get("FACE_AI_TOKENS", "")
        tokens: dict[str, Principal] = {}
        for entry in filter(None, (e.strip() for e in raw.split(";"))):
            parts = [p.strip() for p in entry.split("|")]
            if len(parts) < 2 or not parts[0] or not parts[1]:
                continue
            token, user_id = parts[0], parts[1]
            roles = frozenset(
                filter(None, (r.strip() for r in parts[2].split(","))) if len(parts) > 2 else ()
            )
            collections = frozenset(
                filter(None, (c.strip() for c in parts[3].split(","))) if len(parts) > 3 else ()
            )
            tokens[token] = Principal(
                user_id=user_id, roles=roles, allowed_collections=collections
            )
        return cls(tokens=tokens)

    def resolve(self, token: str | None) -> Principal:
        if not token:
            raise Unauthenticated("thiếu bearer token")
        principal = self.tokens.get(token)
        if principal is None:
            # Không phân biệt "token sai" với "token không tồn tại" để tránh dò token.
            raise Unauthenticated("token không hợp lệ")
        return principal

    @property
    def is_empty(self) -> bool:
        return not self.tokens


# --------------------------------------------------------------------------- #
# Rate limit
# --------------------------------------------------------------------------- #
class SlidingWindowRateLimiter:
    """Rate limit theo principal, cửa sổ trượt 60s, in-process.

    Giới hạn đã biết: in-process nên với nhiều replica thì hạn mức thực tế là
    ``limit * số_replica``. Production dùng Redis. Được viết ra ở đây để hành vi
    429 là thật và test được, chứ không phải TODO.
    """

    def __init__(self, limit_per_minute: int) -> None:
        self._limit = limit_per_minute
        self._window = 60.0
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, *, now: float | None = None) -> None:
        if self._limit <= 0:
            return
        now = time.monotonic() if now is None else now
        with self._lock:
            bucket = self._hits.setdefault(key, [])
            cutoff = now - self._window
            bucket[:] = [t for t in bucket if t > cutoff]
            if len(bucket) >= self._limit:
                retry_after = round(bucket[0] + self._window - now, 2)
                raise RateLimited(
                    f"vượt hạn mức {self._limit} request/phút",
                    details={"retry_after_seconds": max(retry_after, 0.01), "limit": self._limit},
                )
            bucket.append(now)
