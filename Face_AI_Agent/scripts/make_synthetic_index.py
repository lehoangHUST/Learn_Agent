"""Tạo một index thử nghiệm bằng dữ liệu tổng hợp - không dùng ảnh người thật.

Vì sao script này tồn tại: để thử hệ thống (và để hiệu chuẩn thử) cần một tập đã
đăng ký, nhưng không nên lấy ảnh người thật làm dữ liệu demo. Script sinh embedding
tổng hợp **có cấu trúc đúng**:

* mỗi "người" là một vector đơn vị xác định theo chuỗi identity,
* các ảnh khác nhau của cùng người = vector đó cộng nhiễu nhỏ (``--jitter``), cho
  cosine cao - mô phỏng đúng mức đo được trên backend thật (~0.995 cho cùng một
  khuôn mặt ở hai tỷ lệ),
* người khác nhau cho vector gần vuông góc, cosine ~ 0.

Nhờ vậy ngưỡng, margin và ``match_status`` được thử trên dữ liệu có cấu trúc thật,
chỉ là không phải người thật.

Ví dụ::

    python scripts/make_synthetic_index.py --subjects 50 --images-per-subject 3
    python scripts/make_synthetic_index.py --subjects 10 --collection demo --reset

CẢNH BÁO: ngưỡng hiệu chuẩn trên dữ liệu tổng hợp **không** dùng được cho dữ liệu
thật. Phân bố similarity của vector ngẫu nhiên khác hoàn toàn phân bố của khuôn mặt
thật. Dùng index này để kiểm tra *hệ thống*, không phải để chốt *ngưỡng*.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_settings
from app.logging_utils import configure_logging
from app.repositories.base import FaceRecord
from app.repositories.faiss_repository import FaissFaceRepository
from app.security import SYSTEM_PRINCIPAL
from app.vision.fake import identity_vector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, default=20)
    parser.add_argument("--images-per-subject", type=int, default=2)
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.1,
        help="độ lệch giữa các ảnh của cùng một người (0.1 -> cos ~ 0.995)",
    )
    parser.add_argument("--collection", default=None)
    parser.add_argument("--reset", action="store_true", help="xoá index cũ trước khi nạp")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    settings = load_settings(args.config)
    configure_logging(settings.observability.log_level, json_logs=False)

    index_dir = settings.resolve_path(settings.vector_store.index_dir)
    collection = args.collection or settings.vector_store.collection

    if args.reset and index_dir.exists():
        shutil.rmtree(index_dir)
        print(f"Đã xoá {index_dir}")

    repo = FaissFaceRepository(
        index_dir,
        dim=settings.vision.embedding_dim,
        embedding_space=settings.vision.embedding_space,
    )

    records: list[FaceRecord] = []
    now = time.time()
    for s in range(args.subjects):
        subject_id = f"synthetic-subject-{s:03d}"
        for i in range(args.images_per_subject):
            # Ảnh đầu của mỗi người là vector "gốc" (jitter=0); các ảnh sau lệch nhẹ.
            jitter = 0.0 if i == 0 else args.jitter
            records.append(
                FaceRecord(
                    face_id=f"synthetic-{s:03d}-{i:02d}",
                    subject_id=subject_id,
                    # Cùng subject_id, khác variant -> nhiều ảnh của cùng một người.
                    embedding=identity_vector(
                        subject_id,
                        settings.vision.embedding_dim,
                        jitter=jitter,
                        variant=i,
                    ),
                    embedding_space=settings.vision.embedding_space,
                    collection=collection,
                    owner_id="system",
                    metadata={
                        "synthetic": True,
                        "sample_index": i,
                        "note": "dữ liệu tổng hợp, không phải người thật",
                    },
                    created_at=now,
                    consent_ref="synthetic://no-real-person",
                )
            )

    repo.register(records, principal=SYSTEM_PRINCIPAL)
    stats = repo.stats(collection)

    print(f"Đã nạp {len(records)} vector tổng hợp vào collection {collection!r}")
    print(f"  tổng khuôn mặt : {stats.total_faces}")
    print(f"  tổng người     : {stats.total_subjects}")
    print(f"  không gian     : {stats.embedding_space} / {stats.dim}-D / {stats.metric}")
    print(f"  thư mục index  : {index_dir}")
    print(
        "\nLƯU Ý: đây là dữ liệu tổng hợp. Dùng để kiểm tra hệ thống, KHÔNG dùng để "
        "chốt ngưỡng so khớp cho dữ liệu thật."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
