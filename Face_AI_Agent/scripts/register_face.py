"""Đăng ký một khuôn mặt vào index từ dòng lệnh.

Dùng đúng ``register_face_tool`` mà endpoint HTTP dùng, nên hành vi (validate ảnh,
chọn khuôn mặt, kiểm tra không gian embedding, ghi log thiếu consent) giống hệt
nhau - không có hai đường code để lệch nhau.

Ví dụ::

    python scripts/register_face.py --image alice.jpg --subject-id alice \\
        --consent-ref "consent://hr/2026-01-15/alice" --metadata dept=eng

Lưu ý về dữ liệu: chỉ đăng ký ảnh mà bạn có cơ sở pháp lý và sự đồng ý để xử lý.
``--consent-ref`` không bắt buộc về mặt kỹ thuật nhưng thiếu nó sẽ sinh một WARNING
trong log - để việc đăng ký dữ liệu sinh trắc học không có cơ sở đồng ý là thứ
nhìn thấy được, không phải thứ lặng lẽ xảy ra.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_settings
from app.dependencies import build_deps
from app.errors import FaceAIError
from app.logging_utils import configure_logging
from app.security import SYSTEM_PRINCIPAL
from app.tools.search import RegisterFaceInput, register_face_tool


def parse_metadata(pairs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--metadata cần dạng key=value, nhận được {pair!r}")
        key, _, value = pair.partition("=")
        out[key.strip()] = value.strip()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--collection", default=None)
    parser.add_argument(
        "--face-index",
        type=int,
        default=None,
        help="bắt buộc nếu ảnh có nhiều khuôn mặt - hệ thống không tự chọn",
    )
    parser.add_argument("--metadata", nargs="*", default=[], help="key=value ...")
    parser.add_argument("--consent-ref", default=None)
    parser.add_argument(
        "--visibility", choices=["collection", "private"], default="collection"
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    settings = load_settings(args.config)
    configure_logging(settings.observability.log_level, json_logs=False)
    deps = build_deps(settings)

    if not args.image.exists():
        raise SystemExit(f"không có file {args.image}")

    payload = RegisterFaceInput(
        image_base64=base64.b64encode(args.image.read_bytes()).decode("ascii"),
        subject_id=args.subject_id,
        collection=args.collection,
        face_index=args.face_index,
        metadata={**parse_metadata(args.metadata), "source_file": args.image.name},
        consent_ref=args.consent_ref,
        visibility=args.visibility,
    )

    try:
        result = register_face_tool(payload, deps=deps, principal=SYSTEM_PRINCIPAL)
    except FaceAIError as exc:
        print(f"THẤT BẠI [{exc.code}] {exc.message}")
        if exc.details:
            print(json.dumps(exc.details, ensure_ascii=False, indent=2))
        return 1
    finally:
        deps.close()

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["has_consent_ref"]:
        print(
            "\nCẢNH BÁO: đăng ký không có --consent-ref. Hãy ghi lại cơ sở đồng ý "
            "cho dữ liệu sinh trắc học này."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
