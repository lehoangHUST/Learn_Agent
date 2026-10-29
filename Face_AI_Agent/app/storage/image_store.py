"""Kho ảnh tạm trên đĩa.

Mặc định hệ thống **không** lưu ảnh truy vấn: ảnh mặt người là dữ liệu sinh trắc
học, và thứ không được lưu thì không thể bị rò. Kho này chỉ dùng khi:

* ``storage.persist_query_images = true`` (bật tường minh, có chính sách lưu trữ), hoặc
* đăng ký khuôn mặt vào index và cần giữ ảnh gốc làm bằng chứng/đối chiếu.

Mỗi ảnh có ``expires_at``. ``purge_expired`` được gọi lúc startup và có thể gọi
định kỳ; file hết hạn bị xoá khỏi đĩa, không chỉ đánh dấu.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.errors import ResultNotFound
from app.security import Principal

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StoredImage:
    image_ref: str
    mime: str
    size_bytes: int
    sha256: str
    owner_id: str
    created_at: float
    expires_at: float


class LocalImageStore:
    """Kho ảnh trên filesystem.

    Production: đổi sang object storage (S3/GCS/MinIO) với server-side encryption và
    lifecycle rule. Interface giữ nguyên - ``save``/``load``/``delete``/``purge_expired``.
    """

    def __init__(self, image_dir: str | Path, *, ttl_seconds: int = 3600) -> None:
        self._dir = Path(image_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl_seconds

    def _paths(self, image_ref: str) -> tuple[Path, Path]:
        return self._dir / f"{image_ref}.bin", self._dir / f"{image_ref}.json"

    def save(
        self,
        data: bytes,
        *,
        mime: str,
        owner_id: str,
        ttl_seconds: int | None = None,
    ) -> StoredImage:
        image_ref = "img_" + uuid.uuid4().hex[:20]
        now = time.time()
        meta = StoredImage(
            image_ref=image_ref,
            mime=mime,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            owner_id=owner_id,
            created_at=now,
            expires_at=now + (self._ttl if ttl_seconds is None else ttl_seconds),
        )
        blob_path, meta_path = self._paths(image_ref)
        blob_path.write_bytes(data)
        meta_path.write_text(
            json.dumps(
                {
                    "image_ref": meta.image_ref,
                    "mime": meta.mime,
                    "size_bytes": meta.size_bytes,
                    "sha256": meta.sha256,
                    "owner_id": meta.owner_id,
                    "created_at": meta.created_at,
                    "expires_at": meta.expires_at,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        logger.info(
            "image_stored",
            extra={"image_ref": image_ref, "size_bytes": len(data), "mime": mime},
        )
        return meta

    def stat(self, image_ref: str, *, principal: Principal) -> StoredImage:
        _, meta_path = self._paths(image_ref)
        if not meta_path.exists():
            raise ResultNotFound(f"không có ảnh {image_ref!r}")
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        if float(raw["expires_at"]) < time.time():
            self.delete(image_ref)
            raise ResultNotFound(f"ảnh {image_ref!r} đã hết hạn lưu trữ")
        principal.require_owner(raw["owner_id"], what="image")
        return StoredImage(**raw)

    def load(self, image_ref: str, *, principal: Principal) -> tuple[bytes, StoredImage]:
        meta = self.stat(image_ref, principal=principal)
        blob_path, _ = self._paths(image_ref)
        if not blob_path.exists():  # pragma: no cover - file bị xoá tay
            raise ResultNotFound(f"ảnh {image_ref!r} mất phần nội dung")
        return blob_path.read_bytes(), meta

    def delete(self, image_ref: str) -> None:
        for path in self._paths(image_ref):
            path.unlink(missing_ok=True)

    def purge_expired(self, *, now: float | None = None) -> int:
        cutoff = time.time() if now is None else now
        purged = 0
        for meta_path in self._dir.glob("*.json"):
            try:
                raw = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):  # pragma: no cover - meta hỏng
                continue
            if float(raw.get("expires_at", 0)) <= cutoff:
                self.delete(raw.get("image_ref", meta_path.stem))
                purged += 1
        if purged:
            logger.info("images_purged", extra={"purged": purged})
        return purged

    def health(self) -> dict[str, Any]:
        files = list(self._dir.glob("*.bin"))
        return {
            "backend": "local",
            "dir": str(self._dir),
            "stored_images": len(files),
            "ttl_seconds": self._ttl,
        }
