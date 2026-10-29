"""Fixture dùng chung.

Nguyên tắc của bộ test này:

* **Không gọi backend thật.** Mọi test dùng ``FakeFaceEngine``, nhưng engine đó sinh
  vector *có cấu trúc* (cùng identity -> cos ~ 1, khác identity -> cos ~ 0), nên
  logic ngưỡng/margin/match_status được kiểm trên dữ liệu thật sự có ý nghĩa.
* **Không dùng danh tính người thật.** Ảnh test là hình khối do PIL vẽ; embedding
  sinh từ chuỗi identity tổng hợp.
* **Mỗi test một thư mục dữ liệu riêng** (``tmp_path``), nên index/DB không rò giữa
  các test.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Iterator
from typing import Any

import pytest
from PIL import Image, ImageDraw

from app.agent.graph import build_agent_graph
from app.agent.runner import FaceAgentRunner
from app.config import Settings, load_settings
from app.dependencies import AgentDeps, build_deps
from app.repositories.faiss_repository import FaissFaceRepository
from app.security import (
    ROLE_ADMIN,
    ROLE_DELETE,
    ROLE_READ_METADATA,
    ROLE_REGISTER,
    ROLE_SEARCH,
    Principal,
)
from app.vision.fake import FakeFace, FakeFaceEngine, FakeImagePlan

COLLECTION = "test-collection"


# --------------------------------------------------------------------------- #
# Ảnh tổng hợp
# --------------------------------------------------------------------------- #
def make_image_bytes(
    *, width: int = 320, height: int = 240, seed: int = 0, fmt: str = "JPEG"
) -> bytes:
    """Ảnh tổng hợp, khác nhau theo ``seed`` nên sha256 cũng khác nhau."""
    img = Image.new("RGB", (width, height), (240 - (seed * 7) % 60, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        [10 + seed, 10 + seed, 10 + seed + 80, 10 + seed + 90],
        fill=(90 + seed * 5 % 100, 120, 160),
    )
    draw.text((12, 110), f"seed={seed}", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format=fmt, **({"quality": 92} if fmt == "JPEG" else {}))
    return buf.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


@pytest.fixture
def image_bytes() -> bytes:
    return make_image_bytes(seed=1)


@pytest.fixture
def image_b64(image_bytes: bytes) -> str:
    return b64(image_bytes)


# --------------------------------------------------------------------------- #
# Settings & deps
# --------------------------------------------------------------------------- #
@pytest.fixture
def settings(tmp_path) -> Settings:
    cfg = load_settings(config_path="__no_such_config__.yaml", env={})
    cfg.environment = "dev"
    cfg.vision.backend = "fake"
    cfg.vector_store.index_dir = str(tmp_path / "index")
    cfg.vector_store.collection = COLLECTION
    cfg.storage.image_dir = str(tmp_path / "images")
    cfg.storage.results_db_path = str(tmp_path / "results.db")
    cfg.graph.checkpointer = "memory"
    cfg.llm.enabled = False
    cfg.observability.json_logs = False
    cfg.observability.log_level = "WARNING"
    # Ngưỡng cho test: đặt tường minh để test không phụ thuộc giá trị trong YAML.
    cfg.matching.match_threshold = 0.55
    cfg.matching.review_band = 0.05
    cfg.matching.ambiguous_margin = 0.03
    cfg.matching.top_k = 5
    cfg.matching.calibrated = False
    return cfg


@pytest.fixture
def engine(settings: Settings) -> FakeFaceEngine:
    """Engine giả, mặc định mỗi ảnh có đúng một khuôn mặt."""
    return FakeFaceEngine(
        settings.vision,
        default_plan=FakeImagePlan(faces=[FakeFace(identity="synthetic-subject-A")]),
    )


@pytest.fixture
def repository(settings: Settings) -> FaissFaceRepository:
    return FaissFaceRepository(
        settings.resolve_path(settings.vector_store.index_dir),
        dim=settings.vision.embedding_dim,
        embedding_space=settings.vision.embedding_space,
    )


@pytest.fixture
def deps(
    settings: Settings, engine: FakeFaceEngine, repository: FaissFaceRepository
) -> Iterator[AgentDeps]:
    built = build_deps(settings, engine=engine, repository=repository)
    yield built
    built.close()


# --------------------------------------------------------------------------- #
# Principal
# --------------------------------------------------------------------------- #
@pytest.fixture
def principal() -> Principal:
    return Principal(
        user_id="user-alice",
        roles=frozenset({ROLE_SEARCH, ROLE_REGISTER, ROLE_DELETE, ROLE_READ_METADATA}),
        allowed_collections=frozenset({COLLECTION}),
    )


@pytest.fixture
def other_principal() -> Principal:
    """Người dùng khác, cùng collection - dùng để test đọc chéo session."""
    return Principal(
        user_id="user-bob",
        roles=frozenset({ROLE_SEARCH, ROLE_REGISTER, ROLE_READ_METADATA}),
        allowed_collections=frozenset({COLLECTION}),
    )


@pytest.fixture
def admin_principal() -> Principal:
    return Principal(user_id="admin", roles=frozenset({ROLE_ADMIN}))


@pytest.fixture
def no_metadata_principal() -> Principal:
    """Được tìm kiếm nhưng KHÔNG được xem metadata."""
    return Principal(
        user_id="user-carol",
        roles=frozenset({ROLE_SEARCH}),
        allowed_collections=frozenset({COLLECTION}),
    )


# --------------------------------------------------------------------------- #
# Graph / runner
# --------------------------------------------------------------------------- #
@pytest.fixture
def runner(deps: AgentDeps) -> FaceAgentRunner:
    from langgraph.checkpoint.memory import InMemorySaver

    return FaceAgentRunner(build_agent_graph(checkpointer=InMemorySaver()), deps)


# --------------------------------------------------------------------------- #
# Helper nạp dữ liệu
# --------------------------------------------------------------------------- #
@pytest.fixture
def register_subject(deps: AgentDeps, engine: FakeFaceEngine, principal: Principal):
    """Đăng ký một identity tổng hợp vào index và trả về ảnh của nó."""

    def _register(
        subject_id: str,
        *,
        seed: int,
        identity: str | None = None,
        metadata: dict[str, Any] | None = None,
        owner: Principal | None = None,
        visibility: str = "collection",
    ) -> bytes:
        data = make_image_bytes(seed=seed)
        engine.set_plan(data, FakeImagePlan(faces=[FakeFace(identity=identity or subject_id)]))
        analysis = engine.analyze(data)
        deps.search.register_faces(
            analysis.faces,
            subject_id=subject_id,
            principal=owner or principal,
            collection=COLLECTION,
            embedding_space=deps.embedding.embedding_space,
            metadata=metadata or {},
            consent_ref=f"consent://synthetic/{subject_id}",
            visibility=visibility,
        )
        return data

    return _register
