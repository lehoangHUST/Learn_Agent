"""Cấu hình tập trung.

Thứ tự ưu tiên (thấp -> cao): default trong code < configs/config.yaml < biến môi trường.

Cách làm ở đây là *tường minh*: YAML được đọc thành dict, rồi các biến môi trường
được map qua ``_ENV_MAP`` và ghi đè lên dict đó, cuối cùng mới validate bằng pydantic.
Lý do không dùng ``BaseSettings`` với init args: trong pydantic-settings, init args
có ưu tiên CAO HƠN env, nên nếu nhồi YAML vào init thì YAML sẽ thắng env - ngược
với điều ta muốn. Làm tay thì thứ tự ưu tiên là thứ ta viết ra, không phải thứ
ta hy vọng.

Secret (API key, token) CHỈ đọc từ env, không bao giờ nằm trong YAML.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"


class VisionConfig(BaseModel):
    """Backend face detection + embedding (imintv3)."""

    backend: Literal["imint_v3", "fake"] = "imint_v3"
    base_url: str = "http://10.9.3.239:3124"
    extract_feature_path: str = "/imintv3/extract_feature/"
    timeout_seconds: float = 30.0
    connect_timeout_seconds: float = 5.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.5

    # Tham số truyền thẳng xuống API imintv3.
    detect_face: int = 1
    is_checkresize: bool = True

    # Embedding space: 'aligned' -> result.features, 'unaligned' -> result.features_unalign.
    # ĐO ĐƯỢC trên backend thật: cos(aligned, unaligned) ~ 0.04 trên CÙNG một khuôn mặt
    # => đây là hai không gian KHÁC NHAU. Trộn hai loại trong một index cho similarity
    # vô nghĩa. Đổi giá trị này bắt buộc phải build lại toàn bộ index.
    embedding_space: Literal["aligned", "unaligned"] = "aligned"
    embedding_dim: int = 512
    # Backend đã trả vector L2-normalized (đo được: ||v|| = 1.0). Bật cờ này để
    # service tự normalize phòng khi đổi model; không ảnh hưởng nếu đã chuẩn.
    enforce_l2_normalize: bool = True

    # Mặt nhỏ hơn ngưỡng này bị loại trước khi search (đo được: mặt 16px backend
    # không detect nổi; 100px detect tốt).
    min_face_size_px: int = 32


class MatchingConfig(BaseModel):
    """Ngưỡng so khớp.

    CẢNH BÁO: các giá trị dưới đây là PLACEHOLDER CHƯA HIỆU CHUẨN cho model này.
    Chúng được chọn để demo chạy được, KHÔNG phải để quyết định danh tính.
    Chạy ``scripts/calibrate_threshold.py`` trên tập validation của bạn rồi thay số,
    trước khi dùng cho bất kỳ mục đích thật nào.
    """

    metric: Literal["cosine"] = "cosine"
    match_threshold: float = 0.55
    # Khoảng ngay dưới match_threshold -> cần người xem lại thay vì kết luận "không khớp".
    review_band: float = 0.05
    # Nếu top1 - top2 nhỏ hơn mức này thì kết quả mơ hồ, dù top1 vượt ngưỡng.
    ambiguous_margin: float = 0.03
    top_k: int = 5
    calibrated: bool = False
    calibration_note: str = "UNCALIBRATED placeholder. See scripts/calibrate_threshold.py"

    @field_validator("match_threshold", "review_band", "ambiguous_margin")
    @classmethod
    def _cosine_range(cls, v: float) -> float:
        if not -1.0 <= v <= 1.0:
            raise ValueError("tham số cosine phải nằm trong [-1, 1]")
        return v


class VectorStoreConfig(BaseModel):
    backend: Literal["faiss"] = "faiss"
    index_dir: str = "data/index"
    collection: str = "default"


class StorageConfig(BaseModel):
    backend: Literal["local"] = "local"
    image_dir: str = "data/images"
    # Ảnh sinh trắc học là dữ liệu nhạy cảm: giữ càng ngắn càng tốt.
    image_ttl_seconds: int = 3600
    results_db_path: str = "data/results/results.db"
    result_ttl_seconds: int = 86400
    persist_query_images: bool = False


class LLMConfig(BaseModel):
    # enabled=False -> hệ thống chạy hoàn toàn bằng logic xác định (rule-based intent,
    # template answer). Mọi con số vẫn đúng, chỉ là câu trả lời không "mượt".
    enabled: bool = False
    provider: Literal["openai", "none"] = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 800
    timeout_seconds: float = 30.0
    # Rule đi trước; LLM chỉ xử lý phần rule không chắc chắn.
    use_for_intent: bool = True
    use_for_answer: bool = True


class RequestLimitsConfig(BaseModel):
    max_image_bytes: int = 8 * 1024 * 1024
    max_images_per_request: int = 8
    max_image_pixels: int = 50_000_000  # chống decompression bomb
    min_image_side_px: int = 32
    allowed_mime_types: list[str] = Field(
        default_factory=lambda: ["image/jpeg", "image/png", "image/webp", "image/bmp"]
    )
    rate_limit_per_minute: int = 60
    node_timeout_seconds: float = 60.0


class GraphConfig(BaseModel):
    checkpointer: Literal["sqlite", "memory", "none"] = "sqlite"
    checkpoint_db_path: str = "data/results/checkpoints.db"
    recursion_limit: int = 30
    max_node_retries: int = 2


class ObservabilityConfig(BaseModel):
    log_level: str = "INFO"
    json_logs: bool = True
    langfuse_enabled: bool = False


class Settings(BaseModel):
    app_name: str = "face-ai-agent"
    environment: Literal["dev", "staging", "prod"] = "dev"
    vision: VisionConfig = Field(default_factory=VisionConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    limits: RequestLimitsConfig = Field(default_factory=RequestLimitsConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    # --- secrets: chỉ từ env, không bao giờ từ YAML ---
    openai_api_key: str | None = Field(default=None, repr=False)
    langfuse_public_key: str | None = Field(default=None, repr=False)
    langfuse_secret_key: str | None = Field(default=None, repr=False)
    langfuse_host: str | None = None

    def resolve_path(self, relative: str) -> Path:
        """Mọi path trong config đều tương đối so với gốc project."""
        p = Path(relative)
        return p if p.is_absolute() else (REPO_ROOT / p)

    def redacted(self) -> dict[str, Any]:
        """Dump an toàn để log / trả ra /health."""
        data = self.model_dump()
        for key in ("openai_api_key", "langfuse_public_key", "langfuse_secret_key"):
            data[key] = "***set***" if data.get(key) else None
        return data


# --------------------------------------------------------------------------- #
# ENV -> đường dẫn trong cây config. Mỗi dòng là một override tường minh.
# --------------------------------------------------------------------------- #
_ENV_MAP: dict[str, tuple[tuple[str, ...], type]] = {
    "FACE_AI_ENVIRONMENT": (("environment",), str),
    "FACE_AI_VISION_BACKEND": (("vision", "backend"), str),
    "FACE_AI_VISION_BASE_URL": (("vision", "base_url"), str),
    "FACE_AI_VISION_TIMEOUT": (("vision", "timeout_seconds"), float),
    "FACE_AI_VISION_MAX_RETRIES": (("vision", "max_retries"), int),
    "FACE_AI_EMBEDDING_SPACE": (("vision", "embedding_space"), str),
    "FACE_AI_MIN_FACE_SIZE_PX": (("vision", "min_face_size_px"), int),
    "FACE_AI_MATCH_THRESHOLD": (("matching", "match_threshold"), float),
    "FACE_AI_REVIEW_BAND": (("matching", "review_band"), float),
    "FACE_AI_AMBIGUOUS_MARGIN": (("matching", "ambiguous_margin"), float),
    "FACE_AI_TOP_K": (("matching", "top_k"), int),
    "FACE_AI_MATCHING_CALIBRATED": (("matching", "calibrated"), bool),
    "FACE_AI_INDEX_DIR": (("vector_store", "index_dir"), str),
    "FACE_AI_COLLECTION": (("vector_store", "collection"), str),
    "FACE_AI_IMAGE_DIR": (("storage", "image_dir"), str),
    "FACE_AI_IMAGE_TTL": (("storage", "image_ttl_seconds"), int),
    "FACE_AI_RESULTS_DB": (("storage", "results_db_path"), str),
    "FACE_AI_RESULT_TTL": (("storage", "result_ttl_seconds"), int),
    "FACE_AI_PERSIST_QUERY_IMAGES": (("storage", "persist_query_images"), bool),
    "FACE_AI_LLM_ENABLED": (("llm", "enabled"), bool),
    "FACE_AI_LLM_PROVIDER": (("llm", "provider"), str),
    "FACE_AI_LLM_MODEL": (("llm", "model"), str),
    "FACE_AI_LLM_TEMPERATURE": (("llm", "temperature"), float),
    "FACE_AI_MAX_IMAGE_BYTES": (("limits", "max_image_bytes"), int),
    "FACE_AI_MAX_IMAGES_PER_REQUEST": (("limits", "max_images_per_request"), int),
    "FACE_AI_RATE_LIMIT_PER_MINUTE": (("limits", "rate_limit_per_minute"), int),
    "FACE_AI_CHECKPOINTER": (("graph", "checkpointer"), str),
    "FACE_AI_CHECKPOINT_DB": (("graph", "checkpoint_db_path"), str),
    "FACE_AI_LOG_LEVEL": (("observability", "log_level"), str),
    "FACE_AI_JSON_LOGS": (("observability", "json_logs"), bool),
    "FACE_AI_LANGFUSE_ENABLED": (("observability", "langfuse_enabled"), bool),
    # secrets
    "OPENAI_API_KEY": (("openai_api_key",), str),
    "LANGFUSE_PUBLIC_KEY": (("langfuse_public_key",), str),
    "LANGFUSE_SECRET_KEY": (("langfuse_secret_key",), str),
    "LANGFUSE_HOST": (("langfuse_host",), str),
}

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _coerce(raw: str, typ: type) -> Any:
    if typ is bool:
        low = raw.strip().lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise ValueError("khong parse duoc boolean tu " + repr(raw))
    return typ(raw)


def _set_in(tree: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node = tree
    for part in path[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):  # pragma: no cover - config sai cấu trúc
            raise TypeError("config path " + ".".join(path) + " bi chan boi gia tri khong phai dict")
    node[path[-1]] = value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_settings(
    config_path: str | Path | None = None,
    *,
    env: dict[str, str] | None = None,
) -> Settings:
    """Dựng Settings từ YAML + env.

    Thuần và dễ test: ``env`` truyền vào được nên test không cần monkeypatch os.environ.
    """
    env = dict(os.environ if env is None else env)

    path = Path(config_path) if config_path else Path(env.get("FACE_AI_CONFIG", DEFAULT_CONFIG_PATH))
    raw: dict[str, Any] = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise TypeError(str(path) + " phai la mapping o cap cao nhat")
        raw = loaded

    overrides: dict[str, Any] = {}
    for env_name, (dotted, typ) in _ENV_MAP.items():
        if env_name in env and env[env_name] != "":
            _set_in(overrides, dotted, _coerce(env[env_name], typ))

    merged = _deep_merge(raw, overrides)
    return Settings.model_validate(merged)
