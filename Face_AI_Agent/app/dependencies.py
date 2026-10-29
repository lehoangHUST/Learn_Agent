"""Dependency injection: dựng và gắn kết mọi thành phần ở một chỗ.

Module này không import FastAPI và không import LangGraph. Nó chỉ biết config và
các service. Nhờ vậy:

* ``app/agent`` chạy được trong test mà không cần HTTP server,
* ``app/api`` chỉ cần gọi ``get_deps()``,
* test thay engine thật bằng ``FakeFaceEngine`` qua ``build_deps(settings, engine=...)``
  mà không monkeypatch gì cả.

Đây cũng là nơi duy nhất quyết định "backend nào được dùng", theo ``settings``.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from typing import Any

from app.config import Settings, load_settings
from app.repositories.base import BaseFaceSearchRepository
from app.repositories.faiss_repository import FaissFaceRepository
from app.repositories.result_store import SqliteResultStore
from app.security import SlidingWindowRateLimiter, TokenPrincipalStore
from app.services.detection_service import FaceDetectionService
from app.services.embedding_service import FaceEmbeddingService
from app.services.search_service import FaceSearchService
from app.storage.image_store import LocalImageStore
from app.vision.base import BaseFaceEngine
from app.vision.fake import FakeFaceEngine
from app.vision.imint_v3 import ImintV3FaceEngine

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AgentDeps:
    """Bộ dependency mà node của graph được phép dùng (qua ``FaceAgentContext.deps``)."""

    settings: Settings
    engine: BaseFaceEngine
    repository: BaseFaceSearchRepository
    detection: FaceDetectionService
    embedding: FaceEmbeddingService
    search: FaceSearchService
    results: SqliteResultStore
    images: LocalImageStore
    rate_limiter: SlidingWindowRateLimiter
    tokens: TokenPrincipalStore
    llm: Any | None = None  # app.agent.llm.LLMClient | None - import lười để tránh vòng

    def health(self) -> dict[str, Any]:
        return {
            "vision": self.engine.health(),
            "vector_store": self.repository.health(),
            "result_store": self.results.health(),
            "image_store": self.images.health(),
            "llm": (
                self.llm.health()
                if self.llm is not None
                else {"enabled": False, "note": "chạy ở chế độ xác định, không gọi LLM"}
            ),
            "matching": {
                "metric": self.settings.matching.metric,
                "match_threshold": self.settings.matching.match_threshold,
                "calibrated": self.settings.matching.calibrated,
                "note": self.settings.matching.calibration_note,
            },
        }

    def close(self) -> None:
        close = getattr(self.engine, "close", None)
        if callable(close):
            close()


def build_engine(settings: Settings) -> BaseFaceEngine:
    if settings.vision.backend == "fake":
        logger.warning(
            "vision_backend_fake",
            extra={"note": "engine giả - chỉ dùng cho test/offline, không dùng ở prod"},
        )
        return FakeFaceEngine(settings.vision)
    return ImintV3FaceEngine(settings.vision)


def build_deps(
    settings: Settings | None = None,
    *,
    engine: BaseFaceEngine | None = None,
    repository: BaseFaceSearchRepository | None = None,
) -> AgentDeps:
    """Dựng toàn bộ object graph. ``engine``/``repository`` cho phép test tiêm bản giả."""
    settings = settings or load_settings()

    engine = engine or build_engine(settings)
    repository = repository or FaissFaceRepository(
        settings.resolve_path(settings.vector_store.index_dir),
        dim=settings.vision.embedding_dim,
        embedding_space=settings.vision.embedding_space,
    )

    results = SqliteResultStore(
        settings.resolve_path(settings.storage.results_db_path),
        default_ttl_seconds=settings.storage.result_ttl_seconds,
    )
    images = LocalImageStore(
        settings.resolve_path(settings.storage.image_dir),
        ttl_seconds=settings.storage.image_ttl_seconds,
    )

    llm = None
    if settings.llm.enabled:
        from app.agent.llm import LLMClient  # import lười: không bắt buộc cài khi tắt LLM

        llm = LLMClient.create(settings)

    deps = AgentDeps(
        settings=settings,
        engine=engine,
        repository=repository,
        detection=FaceDetectionService(engine, settings.limits),
        embedding=FaceEmbeddingService(engine, settings.vision),
        search=FaceSearchService(repository, settings.matching),
        results=results,
        images=images,
        rate_limiter=SlidingWindowRateLimiter(settings.limits.rate_limit_per_minute),
        tokens=TokenPrincipalStore.from_env(),
        llm=llm,
    )

    if not settings.matching.calibrated:
        logger.warning(
            "matching_threshold_uncalibrated",
            extra={
                "match_threshold": settings.matching.match_threshold,
                "note": settings.matching.calibration_note,
            },
        )
    return deps


def build_checkpointer(settings: Settings) -> Any | None:
    """Checkpointer của LangGraph.

    SQLite cho demo/1 instance. ``check_same_thread=False`` vì endpoint FastAPI sync
    chạy trong threadpool nên saver bị dùng từ nhiều thread; SqliteSaver tự khoá
    quanh mỗi thao tác.

    Nhiều instance -> đổi sang ``PostgresSaver`` (gói ``langgraph-checkpoint-postgres``).
    Interface của saver giống nhau nên chỉ đổi đúng hàm này.
    """
    mode = settings.graph.checkpointer
    if mode == "none":
        return None
    if mode == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()

    from langgraph.checkpoint.sqlite import SqliteSaver

    path = settings.resolve_path(settings.graph.checkpoint_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)


# --------------------------------------------------------------------------- #
# Singleton cho process API
# --------------------------------------------------------------------------- #
_deps: AgentDeps | None = None
_agent: Any | None = None


def get_deps() -> AgentDeps:
    global _deps
    if _deps is None:
        _deps = build_deps()
    return _deps


def get_agent() -> Any:
    """Graph đã compile, dùng chung cho cả process.

    Compile một lần: ``compile()`` kiểm tra topology và dựng runtime, không có lý do
    làm lại mỗi request.
    """
    global _agent
    if _agent is None:
        from app.agent.graph import build_agent_graph

        deps = get_deps()
        _agent = build_agent_graph(checkpointer=build_checkpointer(deps.settings))
    return _agent


def reset_singletons() -> None:
    """Dùng trong test để mỗi test có state sạch."""
    global _deps, _agent
    if _deps is not None:
        _deps.close()
    _deps = None
    _agent = None
