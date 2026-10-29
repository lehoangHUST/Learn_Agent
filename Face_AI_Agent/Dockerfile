# Face AI Agent - image CPU.
#
# Vì sao không có stage GPU trong file này: service này KHÔNG chạy model. Detection
# và embedding do backend imintv3 đảm nhiệm qua HTTP; ở đây chỉ có FAISS (CPU là đủ
# cho tới cỡ 10^6 vector với IndexFlatIP) và logic điều phối. Nhờ vậy image nhẹ,
# không cần CUDA, và chạy được ở mọi nơi.
#
# Nếu sau này đưa model vào trong service (SCRFD/ArcFace chạy local) thì lúc đó mới
# cần base image CUDA - xem ghi chú ở cuối file và phần GPU trong README.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# libgomp1: faiss-cpu cần OpenMP runtime. Thiếu nó thì `import faiss` lỗi
# "libgomp.so.1: cannot open shared object file" - lỗi chỉ xuất hiện trong container,
# không xuất hiện khi dev trên máy.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

# Cài dependency trước, copy code sau: layer cache chỉ bị huỷ khi requirements đổi.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY configs ./configs
COPY scripts ./scripts
COPY pyproject.toml ./

# Chạy bằng user không phải root. Thư mục data phải thuộc user này vì index FAISS,
# SQLite checkpoint và result store đều được ghi lúc chạy.
RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /app/data/index /app/data/images /app/data/results \
 && chown -R appuser:appuser /app/data
USER appuser

ENV FACE_AI_CONFIG=/app/configs/config.yaml

EXPOSE 8000

# Health check gọi endpoint thật. Lưu ý: /health trả "degraded" (vẫn HTTP 200) khi
# index rỗng hoặc ngưỡng chưa hiệu chuẩn - đó là trạng thái vận hành cần biết, không
# phải lý do để orchestrator giết container.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/v1/health || exit 1

# Một worker: FAISS index và SQLite checkpoint đều nằm trong process/trên file cục
# bộ. Muốn scale ra nhiều worker/replica thì phải đổi sang vector DB ngoài và
# PostgresSaver trước - xem phần "Scale" trong README.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# --- Biến thể GPU (chỉ cần khi đưa model vào trong service) ---------------------
# FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
# ... cài python3.12, rồi `pip install faiss-gpu` thay cho faiss-cpu.
# Chạy với `--gpus all` (xem docker-compose.gpu.yml).
