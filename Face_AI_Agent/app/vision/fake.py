"""Engine giả, xác định (deterministic) - dùng cho unit test và chạy offline.

Vì sao cần: unit test không được phụ thuộc một backend GPU trong mạng LAN. Nhưng
test cũng không được dùng mock "trả về gì cũng được", vì khi đó test không chứng
minh được logic matching là đúng. Engine này sinh vector **có ý nghĩa**:

* cùng ``identity`` -> vector gần như trùng nhau (cos ~ 1.0),
* khác ``identity`` -> vector gần vuông góc (cos ~ 0),

nên ngưỡng, margin và match_status được test trên dữ liệu có cấu trúc thật, chỉ là
không phải mặt người thật. Đúng yêu cầu "dữ liệu tổng hợp, không dùng danh tính
người thật để test".

Engine mô phỏng cả các hành vi *kỳ* của backend thật: không có mặt thì trả list
rỗng kèm ``message="Error"``, và không bao giờ có confidence.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.config import VisionConfig
from app.vision.base import DetectedFace, FaceAnalysis, FaceBox
from app.vision.preprocessing import mark_small_faces, order_faces, parse_landmarks


def identity_vector(
    identity: str,
    dim: int = 512,
    *,
    jitter: float = 0.0,
    variant: int = 0,
) -> np.ndarray:
    """Vector đơn vị xác định theo ``identity``.

    * ``jitter = 0``  -> vector "gốc" của người đó.
    * ``jitter > 0``  -> một *ảnh khác* của **cùng** người đó: cùng vector gốc cộng
      một nhiễu nhỏ. ``jitter=0.1`` cho cos ~ 0.995 với vector gốc - đúng mức đo
      được trên backend thật giữa hai ảnh của cùng một khuôn mặt ở hai tỷ lệ.
    * ``variant``     -> chọn nhiễu nào. Hai ảnh khác nhau của cùng một người thì
      cùng ``identity`` nhưng khác ``variant``.

    Cẩn thận chỗ này: muốn tạo "ảnh thứ hai của cùng một người" thì phải giữ
    nguyên ``identity`` và đổi ``variant``. Đổi chuỗi ``identity`` (ví dụ thêm hậu
    tố ``#1``) sẽ sinh ra một **người hoàn toàn khác**, vì vector gốc được seed từ
    chính chuỗi đó - và tập dữ liệu "cùng người" sẽ có phân bố similarity giống hệt
    tập "khác người", làm mọi phép hiệu chuẩn thành vô nghĩa.
    """
    seed = int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    base = rng.normal(size=dim).astype(np.float32)
    base /= np.linalg.norm(base)
    if jitter > 0:
        noise_seed = int.from_bytes(
            hashlib.sha256(f"{identity}:jitter:{variant}".encode()).digest()[:8], "big"
        )
        noise = np.random.default_rng(noise_seed).normal(size=dim).astype(np.float32)
        noise /= np.linalg.norm(noise)
        base = base + jitter * noise
        base /= np.linalg.norm(base)
    return base.astype(np.float32)


@dataclass(slots=True)
class FakeFace:
    """Một mặt được kịch bản hoá cho một ảnh."""

    identity: str
    box: tuple[float, float, float, float] = (10.0, 10.0, 110.0, 120.0)
    jitter: float = 0.0
    # Hai ảnh khác nhau của CÙNG một người: cùng ``identity``, khác ``variant``.
    variant: int = 0
    # None -> engine tự sinh từ identity. Đặt tay khi test cần vector cụ thể.
    embedding: np.ndarray | None = None


@dataclass(slots=True)
class FakeImagePlan:
    faces: list[FakeFace] = field(default_factory=list)
    message: str | None = None
    # Lỗi để inject - dùng cho test "một node bị lỗi giữa graph".
    raise_error: Exception | None = None
    image_width: int = 640
    image_height: int = 480


class FakeFaceEngine:
    """Hiện thực cùng bộ interface như ``ImintV3FaceEngine``."""

    name = "fake"

    def __init__(
        self,
        config: VisionConfig,
        plans: dict[str, FakeImagePlan] | None = None,
        *,
        default_plan: FakeImagePlan | None = None,
    ) -> None:
        self._config = config
        # key = sha256 của bytes ảnh -> kịch bản
        self._plans: dict[str, FakeImagePlan] = plans or {}
        self._default_plan = default_plan
        self.call_count = 0

    # ---- điều khiển từ test ---------------------------------------- #
    def set_plan(self, image_bytes: bytes, plan: FakeImagePlan) -> str:
        digest = hashlib.sha256(image_bytes).hexdigest()
        self._plans[digest] = plan
        return digest

    def set_default_plan(self, plan: FakeImagePlan | None) -> None:
        self._default_plan = plan

    # ---- BaseFaceEmbedder ------------------------------------------ #
    @property
    def embedding_dim(self) -> int:
        return self._config.embedding_dim

    @property
    def embedding_space(self) -> str:
        return self._config.embedding_space

    # ---- BaseFaceEngine -------------------------------------------- #
    def detect(self, image_bytes: bytes) -> FaceAnalysis:
        return self.analyze(image_bytes)

    def embed(self, image_bytes: bytes) -> FaceAnalysis:
        return self.analyze(image_bytes)

    def analyze(self, image_bytes: bytes) -> FaceAnalysis:
        self.call_count += 1
        started = time.perf_counter()
        digest = hashlib.sha256(image_bytes).hexdigest()
        plan = self._plans.get(digest, self._default_plan)

        if plan is None:
            # Không có kịch bản -> mặc định một mặt, identity suy ra từ nội dung ảnh.
            plan = FakeImagePlan(faces=[FakeFace(identity=f"auto:{digest[:12]}")])

        if plan.raise_error is not None:
            raise plan.raise_error

        warnings: list[str] = []
        faces: list[DetectedFace] = []
        for idx, spec in enumerate(plan.faces):
            vec = (
                spec.embedding
                if spec.embedding is not None
                else identity_vector(
                    spec.identity,
                    self._config.embedding_dim,
                    jitter=spec.jitter,
                    variant=spec.variant,
                )
            )
            vec = np.asarray(vec, dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if self._config.enforce_l2_normalize and norm > 0 and abs(norm - 1.0) > 1e-6:
                vec = vec / norm

            box = FaceBox.from_sequence(spec.box)
            # landmark tổng hợp, đặt theo tỷ lệ trong box cho giống layout 5 điểm thật
            w, h = box.width, box.height
            flat = [
                box.x1 + 0.30 * w, box.y1 + 0.35 * h,
                box.x1 + 0.70 * w, box.y1 + 0.35 * h,
                box.x1 + 0.50 * w, box.y1 + 0.55 * h,
                box.x1 + 0.35 * w, box.y1 + 0.75 * h,
                box.x1 + 0.65 * w, box.y1 + 0.75 * h,
            ]
            faces.append(
                DetectedFace(
                    face_index=idx,
                    box=box,
                    landmarks=parse_landmarks(flat),
                    confidence=None,  # giống backend thật: không có confidence
                    confidence_source="unavailable_from_backend",
                    embedding=vec,
                    embedding_space=self.embedding_space,
                )
            )

        faces = order_faces(faces)
        warnings.extend(mark_small_faces(faces, self._config.min_face_size_px))

        message = plan.message or ("success" if faces else "Error")
        if not faces and plan.message is None:
            warnings.append(
                "backend báo message='Error' khi không phát hiện khuôn mặt nào "
                "(mô phỏng hành vi đã biết của imintv3)"
            )

        return FaceAnalysis(
            faces=faces,
            image_width=plan.image_width,
            image_height=plan.image_height,
            backend=self.name,
            backend_message=message,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            warnings=warnings,
        )

    def health(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "reachable": True,
            "embedding_space": self.embedding_space,
            "embedding_dim": self.embedding_dim,
            "detector_confidence_available": False,
            "calls": self.call_count,
        }

    def close(self) -> None:  # pragma: no cover - không giữ tài nguyên
        return None
