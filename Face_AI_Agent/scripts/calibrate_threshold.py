"""Hiệu chuẩn ngưỡng so khớp trên tập validation của bạn.

Vì sao bắt buộc phải làm bước này
---------------------------------
``matching.match_threshold`` trong config là một con số **placeholder**. Ngưỡng đúng
phụ thuộc vào: model embedding cụ thể, chất lượng ảnh (độ nét, ánh sáng, góc, độ
phân giải), dân số người dùng, và mức sai nào bạn chấp nhận được. Không có con số
nào "đúng chung" - một ngưỡng mượn từ bài báo hay từ project khác là một ngưỡng
chưa hiệu chuẩn.

Hai loại sai, và chúng đánh đổi với nhau
----------------------------------------
* **FMR** (False Match Rate) - tỷ lệ hai người KHÁC nhau bị coi là khớp. Ngưỡng
  càng thấp, FMR càng cao. Đây là loại sai gây hậu quả cho người bị nhận dạng nhầm.
* **FNMR** (False Non-Match Rate) - tỷ lệ hai ảnh của CÙNG người bị coi là không
  khớp. Ngưỡng càng cao, FNMR càng cao. Đây là loại sai gây bất tiện.

Chọn ngưỡng = chọn điểm làm việc trên đường đánh đổi đó, theo *hậu quả* của từng
loại sai trong hệ thống của bạn - không phải theo con số nào đẹp. Với các quyết định
có hậu quả nghiêm trọng, hãy chọn FMR thấp và chấp nhận FNMR cao hơn, rồi để người
xem lại những trường hợp trong dải ``needs_review``.

**EER** (Equal Error Rate, điểm FMR = FNMR) được in ra như một mốc tham khảo để so
sánh các model, **không** phải để làm ngưỡng vận hành: nó cân bằng hai loại sai như
thể chúng có hậu quả như nhau, điều gần như không bao giờ đúng.

Giới hạn thống kê cần biết
--------------------------
Muốn đo FMR ở mức 1e-4 một cách có ý nghĩa thì cần ở mức 10^4-10^5 cặp impostor.
Script in ra số cặp thực tế và nói rõ mức FMR nào là *đo được* với tập của bạn;
những mức thấp hơn thế chỉ là nội suy từ không đủ dữ liệu.

Cách dùng
---------
Từ cây thư mục ``subject_id/*.jpg`` (mỗi người cần >= 2 ảnh để có cặp genuine)::

    python scripts/calibrate_threshold.py --dataset ./validation_set

Từ file cặp đã có nhãn (CSV: ``path_a,path_b,label`` với label là ``same``/``diff``)::

    python scripts/calibrate_threshold.py --pairs pairs.csv

Thử nhanh trên dữ liệu tổng hợp (chỉ để xem script chạy, KHÔNG dùng số kết quả)::

    python scripts/calibrate_threshold.py --synthetic 40

Dữ liệu: chỉ dùng ảnh tổng hợp hoặc ảnh có sự đồng ý và quyền sử dụng hợp pháp.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.config import load_settings
from app.dependencies import build_engine
from app.errors import FaceAIError
from app.logging_utils import configure_logging
from app.vision.fake import identity_vector

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TARGET_FMRS = (1e-1, 1e-2, 1e-3, 1e-4)


# --------------------------------------------------------------------------- #
# Lấy embedding
# --------------------------------------------------------------------------- #
def embed_dataset(dataset: Path, engine, max_per_subject: int | None) -> dict[str, list[np.ndarray]]:
    """``{subject_id: [vector, ...]}``. Ảnh không có đúng 1 mặt bị bỏ qua."""
    vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    skipped = 0

    for subject_dir in sorted(p for p in dataset.iterdir() if p.is_dir()):
        images = sorted(p for p in subject_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if max_per_subject:
            images = images[:max_per_subject]
        for image in images:
            try:
                analysis = engine.analyze(image.read_bytes())
            except FaceAIError as exc:
                print(f"  [bỏ qua] {image}: {exc.code}")
                skipped += 1
                continue
            usable = [f for f in analysis.usable_faces if f.embedding is not None]
            if len(usable) != 1:
                print(
                    f"  [bỏ qua] {image}: có {len(usable)} khuôn mặt dùng được "
                    "(cần đúng 1 để hiệu chuẩn)"
                )
                skipped += 1
                continue
            vectors[subject_dir.name].append(usable[0].embedding)

    if skipped:
        print(f"  -> bỏ qua {skipped} ảnh")
    return dict(vectors)


def synthetic_dataset(subjects: int, images_per_subject: int, dim: int) -> dict[str, list[np.ndarray]]:
    # Cùng ``identity``, khác ``variant`` -> nhiều ảnh của CÙNG một người.
    # Đổi chuỗi identity ở đây sẽ sinh ra người khác và làm hiệu chuẩn vô nghĩa.
    return {
        f"synthetic-{s:03d}": [
            identity_vector(f"synthetic-{s:03d}", dim)
            if i == 0
            else identity_vector(f"synthetic-{s:03d}", dim, jitter=0.12, variant=i)
            for i in range(images_per_subject)
        ]
        for s in range(subjects)
    }


# --------------------------------------------------------------------------- #
# Sinh cặp
# --------------------------------------------------------------------------- #
def build_pairs(
    vectors: dict[str, list[np.ndarray]],
    *,
    max_impostor_pairs: int,
    seed: int = 0,
) -> tuple[list[float], list[float]]:
    """Trả về (điểm của cặp cùng người, điểm của cặp khác người).

    Cặp genuine: mọi tổ hợp 2 ảnh trong cùng một người.
    Cặp impostor: lấy mẫu ngẫu nhiên giữa các người khác nhau - lấy *tất cả* sẽ bùng
    nổ theo bình phương số ảnh, mà một mẫu ngẫu nhiên đủ lớn đã cho FMR đáng tin.
    """
    genuine: list[float] = []
    for samples in vectors.values():
        for a, b in itertools.combinations(samples, 2):
            genuine.append(float(a @ b))

    flat = [(subject, vec) for subject, samples in vectors.items() for vec in samples]
    rng = random.Random(seed)
    impostor: list[float] = []
    total_possible = len(flat) * (len(flat) - 1) // 2

    if total_possible <= max_impostor_pairs:
        for (sa, va), (sb, vb) in itertools.combinations(flat, 2):
            if sa != sb:
                impostor.append(float(va @ vb))
    else:
        seen: set[tuple[int, int]] = set()
        attempts = 0
        while len(impostor) < max_impostor_pairs and attempts < max_impostor_pairs * 20:
            attempts += 1
            i, j = rng.randrange(len(flat)), rng.randrange(len(flat))
            if i == j or flat[i][0] == flat[j][0]:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            impostor.append(float(flat[i][1] @ flat[j][1]))

    return genuine, impostor


def pairs_from_csv(path: Path, engine) -> tuple[list[float], list[float]]:
    genuine: list[float] = []
    impostor: list[float] = []
    cache: dict[Path, np.ndarray | None] = {}

    def vector_of(image_path: Path) -> np.ndarray | None:
        if image_path not in cache:
            try:
                analysis = engine.analyze(image_path.read_bytes())
                usable = [f for f in analysis.usable_faces if f.embedding is not None]
                cache[image_path] = usable[0].embedding if len(usable) == 1 else None
            except FaceAIError:
                cache[image_path] = None
        return cache[image_path]

    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 3 or row[0].startswith("#"):
                continue
            left, right, label = Path(row[0].strip()), Path(row[1].strip()), row[2].strip().lower()
            va, vb = vector_of(left), vector_of(right)
            if va is None or vb is None:
                print(f"  [bỏ qua] cặp {left.name} / {right.name}: không lấy được embedding")
                continue
            score = float(va @ vb)
            (genuine if label in ("same", "1", "genuine") else impostor).append(score)
    return genuine, impostor


# --------------------------------------------------------------------------- #
# Tính FMR/FNMR
# --------------------------------------------------------------------------- #
def error_rates(genuine: np.ndarray, impostor: np.ndarray, threshold: float) -> tuple[float, float]:
    fnmr = float(np.mean(genuine < threshold)) if genuine.size else float("nan")
    fmr = float(np.mean(impostor >= threshold)) if impostor.size else float("nan")
    return fmr, fnmr


def threshold_for_fmr(impostor: np.ndarray, target_fmr: float) -> float:
    """Ngưỡng nhỏ nhất sao cho FMR <= target.

    Dùng phân vị của phân bố impostor: ngưỡng tại phân vị (1 - target).
    """
    if impostor.size == 0:
        return float("nan")
    return float(np.quantile(impostor, 1.0 - target_fmr))


def find_eer(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """(ngưỡng, giá trị EER) bằng cách quét trên các điểm score thực tế."""
    candidates = np.unique(np.concatenate([genuine, impostor]))
    best = (float("nan"), float("inf"))
    for threshold in candidates:
        fmr, fnmr = error_rates(genuine, impostor, float(threshold))
        gap = abs(fmr - fnmr)
        if gap < best[1]:
            best = (float(threshold), gap)
    threshold = best[0]
    fmr, fnmr = error_rates(genuine, impostor, threshold)
    return threshold, (fmr + fnmr) / 2.0


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path, help="thư mục subject_id/*.jpg")
    source.add_argument("--pairs", type=Path, help="CSV: path_a,path_b,same|diff")
    source.add_argument(
        "--synthetic", type=int, metavar="N", help="N người tổng hợp (chỉ để thử script)"
    )
    parser.add_argument("--images-per-subject", type=int, default=None)
    parser.add_argument("--max-impostor-pairs", type=int, default=20000)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    settings = load_settings(args.config)
    configure_logging("WARNING", json_logs=False)

    if args.synthetic:
        print(f"Dùng {args.synthetic} người TỔNG HỢP (dữ liệu giả).\n")
        vectors = synthetic_dataset(
            args.synthetic, args.images_per_subject or 3, settings.vision.embedding_dim
        )
        genuine_list, impostor_list = build_pairs(
            vectors, max_impostor_pairs=args.max_impostor_pairs
        )
    else:
        engine = build_engine(settings)
        print(
            f"Backend: {engine.name} | không gian: {settings.vision.embedding_space} "
            f"| {settings.vision.embedding_dim}-D\n"
        )
        try:
            if args.dataset:
                if not args.dataset.is_dir():
                    raise SystemExit(f"{args.dataset} không phải thư mục")
                print("Đang tính embedding...")
                vectors = embed_dataset(args.dataset, engine, args.images_per_subject)
                multi = {s: v for s, v in vectors.items() if len(v) >= 2}
                if not multi:
                    raise SystemExit(
                        "cần ít nhất một người có >= 2 ảnh để có cặp genuine"
                    )
                genuine_list, impostor_list = build_pairs(
                    vectors, max_impostor_pairs=args.max_impostor_pairs
                )
            else:
                print("Đang tính embedding cho các cặp...")
                genuine_list, impostor_list = pairs_from_csv(args.pairs, engine)
        finally:
            close = getattr(engine, "close", None)
            if callable(close):
                close()

    genuine = np.asarray(genuine_list, dtype=np.float64)
    impostor = np.asarray(impostor_list, dtype=np.float64)

    if genuine.size == 0 or impostor.size == 0:
        raise SystemExit(
            f"cần cả hai loại cặp (genuine={genuine.size}, impostor={impostor.size})"
        )

    print("\n" + "=" * 72)
    print("PHÂN BỐ SIMILARITY")
    print("=" * 72)
    print(f"Cặp cùng người (genuine) : n={genuine.size}")
    print(
        f"  min={genuine.min():.4f}  p05={np.quantile(genuine, 0.05):.4f}  "
        f"median={np.median(genuine):.4f}  max={genuine.max():.4f}"
    )
    print(f"Cặp khác người (impostor): n={impostor.size}")
    print(
        f"  min={impostor.min():.4f}  median={np.median(impostor):.4f}  "
        f"p95={np.quantile(impostor, 0.95):.4f}  max={impostor.max():.4f}"
    )

    # Mức FMR thấp nhất còn *đo được* với số cặp impostor hiện có.
    measurable_fmr = 1.0 / impostor.size
    print(f"\nMức FMR thấp nhất còn đo được với {impostor.size} cặp: {measurable_fmr:.1e}")
    print(
        "  (muốn đo FMR 1e-4 có ý nghĩa thì cần cỡ 10^4-10^5 cặp impostor; "
        "các mức dưới ngưỡng này chỉ là nội suy)"
    )

    print("\n" + "=" * 72)
    print("ĐIỂM LÀM VIỆC THEO FMR MỤC TIÊU")
    print("=" * 72)
    print(f"{'FMR mục tiêu':<14}{'ngưỡng':<12}{'FMR thực':<12}{'FNMR':<12}{'đo được?'}")
    rows: list[tuple[float, float, float, float]] = []
    for target in TARGET_FMRS:
        threshold = threshold_for_fmr(impostor, target)
        fmr, fnmr = error_rates(genuine, impostor, threshold)
        reliable = "có" if target >= measurable_fmr else "KHÔNG (thiếu dữ liệu)"
        rows.append((target, threshold, fmr, fnmr))
        print(f"{target:<14.0e}{threshold:<12.4f}{fmr:<12.4f}{fnmr:<12.4f}{reliable}")

    eer_threshold, eer = find_eer(genuine, impostor)
    print(f"\nEER = {eer:.4f} tại ngưỡng {eer_threshold:.4f}")
    print(
        "  EER chỉ để so sánh model. KHÔNG dùng làm ngưỡng vận hành: nó coi hai loại "
        "sai có hậu quả như nhau."
    )

    # Đề xuất: FMR 1e-3 nếu còn đo được, nếu không thì mức thấp nhất đo được.
    usable = [r for r in rows if r[0] >= measurable_fmr]
    chosen = next((r for r in usable if r[0] <= 1e-3), usable[-1] if usable else rows[0])
    target, threshold, fmr, fnmr = chosen
    review_band = max(0.02, round(float(np.std(genuine)) / 2, 4))
    ambiguous_margin = 0.03

    print("\n" + "=" * 72)
    print("ĐOẠN CẤU HÌNH ĐỀ XUẤT (configs/config.yaml)")
    print("=" * 72)
    print("matching:")
    print("  metric: cosine")
    print(f"  match_threshold: {threshold:.4f}   # FMR~{fmr:.4f}, FNMR~{fnmr:.4f}")
    print(f"  review_band: {review_band}")
    print(f"  ambiguous_margin: {ambiguous_margin}")
    print("  top_k: 5")
    print("  calibrated: true")
    print(
        f'  calibration_note: "FMR mục tiêu {target:.0e} trên {genuine.size} cặp '
        f'genuine / {impostor.size} cặp impostor"'
    )

    if args.synthetic:
        print("\n" + "!" * 72)
        print("ĐÂY LÀ DỮ LIỆU TỔNG HỢP. KHÔNG dùng những con số trên cho hệ thống thật.")
        print("Phân bố similarity của vector ngẫu nhiên khác hoàn toàn khuôn mặt thật.")
        print("!" * 72)
    else:
        print(
            "\nTrước khi đặt calibrated: true, hãy kiểm tra tập validation có đại diện "
            "cho dân số và điều kiện ảnh của hệ thống đích. Một ngưỡng hiệu chuẩn trên "
            "tập không đại diện vẫn là một ngưỡng sai."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
