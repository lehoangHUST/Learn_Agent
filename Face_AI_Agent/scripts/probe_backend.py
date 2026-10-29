"""Kiểm tra backend face có đúng những giả định mà hệ thống này dựa vào.

Chạy script này **trước khi** tin vào bất cứ kết quả nào, và chạy lại mỗi khi đổi
backend hoặc đổi model. Nó kiểm chứng từng giả định một, trên dữ liệu thật của
backend:

1. Endpoint gọi được, response có cấu trúc ``message``/``result``.
2. ``features`` có đúng số chiều như ``vision.embedding_dim``.
3. ``features`` đã L2-normalize (nếu không, cosine qua IndexFlatIP sẽ sai).
4. ``features`` và ``features_unalign`` là hai không gian KHÁC nhau.
5. ``boxes`` có mấy giá trị - có confidence hay không.
6. Ảnh không có mặt trả về gì (hành vi "message=Error + list rỗng").
7. Cùng một khuôn mặt ở hai tỷ lệ cho similarity cao (pipeline đúng).

Cách dùng::

    python scripts/probe_backend.py --image path/to/face.jpg
    python scripts/probe_backend.py --image face.jpg --base-url http://10.9.3.239:3124
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

# Chạy được bằng cả `python scripts/x.py` và `python -m scripts.x`: khi chạy trực
# tiếp, Python đặt scripts/ vào sys.path chứ không đặt gốc project.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from app.config import load_settings
from app.vision.imint_v3 import ImintV3FaceEngine

PASS = "  [PASS]"
FAIL = "  [FAIL]"
INFO = "  [info]"


def _encode(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="ảnh có đúng một khuôn mặt")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    settings = load_settings(args.config)
    if args.base_url:
        settings.vision.base_url = args.base_url

    engine = ImintV3FaceEngine(settings.vision)
    failures = 0

    print(f"Backend: {settings.vision.base_url}{settings.vision.extract_feature_path}")
    print(f"Cấu hình: embedding_space={settings.vision.embedding_space} "
          f"dim={settings.vision.embedding_dim}\n")

    # 1. Reachability
    print("1. Gọi được backend?")
    health = engine.health()
    if health.get("reachable"):
        print(f"{PASS} reachable (HTTP {health.get('status_code')})")
    else:
        print(f"{FAIL} không gọi được: {health}")
        return 1

    path = Path(args.image)
    if not path.exists():
        print(f"{FAIL} không có file {path}")
        return 1
    original = Image.open(path)
    analysis = engine.analyze(_encode(original))

    print(f"\n2. Phát hiện được khuôn mặt? (ảnh {original.size[0]}x{original.size[1]})")
    if analysis.face_count == 0:
        print(f"{FAIL} không phát hiện khuôn mặt nào - hãy dùng ảnh khác để probe")
        print(f"{INFO} backend_message={analysis.backend_message!r}")
        return 1
    print(f"{PASS} {analysis.face_count} khuôn mặt, latency {analysis.latency_ms}ms")

    face = analysis.faces[0]

    # 3. Dimension
    print(f"\n3. Số chiều embedding == {settings.vision.embedding_dim}?")
    assert face.embedding is not None
    dim = int(face.embedding.shape[0])
    if dim == settings.vision.embedding_dim:
        print(f"{PASS} {dim}-D")
    else:
        print(f"{FAIL} backend trả {dim}-D; sửa vision.embedding_dim và build lại index")
        failures += 1

    # 4. L2 normalization
    print("\n4. Vector đã L2-normalize? (cần thiết để IndexFlatIP = cosine)")
    norm = float(np.linalg.norm(face.embedding))
    if abs(norm - 1.0) <= 1e-3:
        print(f"{PASS} ||v|| = {norm:.6f}")
    else:
        print(f"{INFO} ||v|| = {norm:.6f} - hệ thống sẽ tự normalize "
              "(enforce_l2_normalize=true), nhưng nên biết")

    # 5. aligned vs unaligned
    print("\n5. 'aligned' và 'unaligned' là hai không gian khác nhau?")
    other_space = "unaligned" if settings.vision.embedding_space == "aligned" else "aligned"
    # Dùng BẢN COPY của config, không mutate bản gốc: ``ImintV3FaceEngine`` giữ một
    # tham chiếu tới config, nên sửa ``settings.vision`` tại chỗ sẽ đổi luôn không
    # gian embedding của engine đã tạo ở trên - và các kiểm tra sau đó so sánh vector
    # của hai không gian khác nhau rồi báo "similarity thấp bất thường".
    other_config = settings.vision.model_copy(update={"embedding_space": other_space})
    other_engine = ImintV3FaceEngine(other_config)
    other = other_engine.analyze(_encode(original))
    if other.faces and other.faces[0].embedding is not None:
        cross = float(face.embedding @ other.faces[0].embedding)
        print(f"{INFO} cos(aligned, unaligned) = {cross:.4f} trên CÙNG một khuôn mặt")
        if abs(cross) < 0.5:
            print(f"{PASS} hai không gian không tương thích - đúng như hệ thống giả định; "
                  "KHÔNG được trộn chúng trong một index")
        else:
            print(f"{INFO} hai không gian khá giống nhau; giả định của hệ thống vẫn an toàn "
                  "(nó luôn chốt theo một không gian)")

    # 6. Confidence
    print("\n6. Detector có trả confidence?")
    if face.confidence is None:
        print(f"{PASS} không có confidence (box chỉ có 4 số) - đúng như hệ thống giả định; "
              "API sẽ trả confidence=null thay vì số bịa")
    else:
        print(f"{INFO} backend CÓ trả confidence = {face.confidence}. Hệ thống sẽ "
              "hiển thị nó; cân nhắc cập nhật các ghi chú 'không có confidence'")

    # 7. Ảnh không có mặt
    print("\n7. Ảnh không có khuôn mặt trả về gì?")
    blank = Image.new("RGB", (320, 240), (128, 128, 128))
    blank_analysis = engine.analyze(_encode(blank))
    if blank_analysis.face_count == 0:
        print(f"{PASS} 0 khuôn mặt, backend_message={blank_analysis.backend_message!r} "
              "(được xử lý như kết quả hợp lệ, không phải lỗi)")
    else:
        print(f"{FAIL} phát hiện {blank_analysis.face_count} 'khuôn mặt' trong ảnh trống - "
              "detector có vấn đề")
        failures += 1

    # 8. Tính nhất quán theo tỷ lệ
    print("\n8. Cùng khuôn mặt ở tỷ lệ khác -> similarity cao?")
    scaled = original.resize((original.size[0] * 2, original.size[1] * 2))
    scaled_analysis = engine.analyze(_encode(scaled))
    if scaled_analysis.faces and scaled_analysis.faces[0].embedding is not None:
        sim = float(face.embedding @ scaled_analysis.faces[0].embedding)
        print(f"{INFO} cos = {sim:.4f}")
        if sim > 0.8:
            print(f"{PASS} pipeline nhất quán")
        else:
            print(f"{FAIL} similarity thấp bất thường cho cùng một khuôn mặt - "
                  "kiểm tra lại alignment/tiền xử lý")
            failures += 1
    else:
        print(f"{FAIL} không phát hiện được khuôn mặt trên ảnh đã scale")
        failures += 1

    print("\n" + "=" * 70)
    if failures:
        print(f"{failures} kiểm tra THẤT BẠI - đừng tin kết quả search cho tới khi sửa xong.")
    else:
        print("Tất cả giả định về backend đều đúng.")
    print("Lưu ý: điều này KHÔNG có nghĩa ngưỡng so khớp đã đúng. "
          "Chạy scripts/calibrate_threshold.py.")
    print("=" * 70)
    engine.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
