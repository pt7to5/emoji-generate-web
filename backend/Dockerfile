FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=9000

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libxcb1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.prod.txt ./requirements.prod.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.1" "torchvision>=0.16" && \
    pip install -r requirements.prod.txt

COPY app ./app
COPY models ./models

EXPOSE 9000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-9000}"]
