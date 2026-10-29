"""Nạp hàng loạt khuôn mặt vào index từ một cây thư mục.

Cấu trúc thư mục mong đợi - mỗi thư mục con là một người::

    dataset/
      alice/
        01.jpg
        02.jpg
      bob/
        01.jpg

Mỗi ảnh nên có đúng một khuôn mặt. Ảnh nhiều mặt bị **bỏ qua** kèm cảnh báo, chứ
không tự chọn mặt nào: đăng ký sai người vào index làm bẩn mọi kết quả search về
sau và rất khó phát hiện.

Ví dụ::

    python scripts/build_index.py --dataset ./dataset --consent-ref "consent://demo"
    python scripts/build_index.py --dataset ./dataset --dry-run

Dữ liệu: chỉ dùng ảnh tổng hợp hoặc ảnh có sự đồng ý và quyền sử dụng hợp pháp.
Để có một index thử nghiệm mà không cần ảnh người thật, dùng
``scripts/make_synthetic_index.py``.
"""

from __future__ import annotations

import argparse
import base64
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_settings
from app.dependencies import build_deps
from app.errors import FaceAIError
from app.logging_utils import configure_logging
from app.security import SYSTEM_PRINCIPAL
from app.tools.search import RegisterFaceInput, register_face_tool

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--consent-ref", default=None)
    parser.add_argument("--limit-per-subject", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="chỉ liệt kê những gì sẽ nạp, không gọi backend và không ghi index",
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    settings = load_settings(args.config)
    configure_logging(settings.observability.log_level, json_logs=False)

    if not args.dataset.is_dir():
        raise SystemExit(f"{args.dataset} không phải thư mục")

    subjects = sorted(p for p in args.dataset.iterdir() if p.is_dir())
    if not subjects:
        raise SystemExit(f"không có thư mục con nào trong {args.dataset}")

    plan: list[tuple[str, Path]] = []
    for subject_dir in subjects:
        images = sorted(
            p for p in subject_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
        )
        if args.limit_per_subject:
            images = images[: args.limit_per_subject]
        if not images:
            print(f"[bỏ qua] {subject_dir.name}: không có ảnh hợp lệ")
            continue
        plan.extend((subject_dir.name, image) for image in images)

    print(f"Sẽ nạp {len(plan)} ảnh của {len({s for s, _ in plan})} người.")
    if args.dry_run:
        for subject, image in plan:
            print(f"  {subject:<24} {image.name}")
        print("\n--dry-run: không gọi backend, không ghi index.")
        return 0

    deps = build_deps(settings)
    collection = args.collection or settings.vector_store.collection
    stats: Counter[str] = Counter()

    try:
        for subject, image in plan:
            try:
                register_face_tool(
                    RegisterFaceInput(
                        image_base64=base64.b64encode(image.read_bytes()).decode("ascii"),
                        subject_id=subject,
                        collection=collection,
                        metadata={"source_file": image.name},
                        consent_ref=args.consent_ref,
                    ),
                    deps=deps,
                    principal=SYSTEM_PRINCIPAL,
                )
                stats["registered"] += 1
                print(f"  [ok]      {subject:<24} {image.name}")
            except FaceAIError as exc:
                stats[exc.code] += 1
                print(f"  [{exc.code}] {subject:<24} {image.name}: {exc.message}")

        print("\nTổng kết:")
        for key, count in stats.most_common():
            print(f"  {key:<32} {count}")
        print("\nIndex:", deps.search.stats(collection))
        if not settings.matching.calibrated:
            print(
                "\nBƯỚC TIẾP THEO: ngưỡng so khớp vẫn chưa được hiệu chuẩn. "
                "Chạy scripts/calibrate_threshold.py trước khi dùng kết quả search "
                "cho bất kỳ quyết định nào."
            )
    finally:
        deps.close()

    return 0 if stats["registered"] else 1


if __name__ == "__main__":
    sys.exit(main())
