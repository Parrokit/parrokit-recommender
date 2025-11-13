# Dockerfile (multi-stage: builder + runtime)

# ----------------------------
# 1) Builder Stage (빌드 전용)
# ----------------------------
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 빌드에만 필요한 패키지 설치 (컴파일러 등)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements 먼저 복사해서 캐시 최대 활용
COPY requirements.txt .

# wheel(빌드 결과물)만 만들어서 /wheels 에 모아두기
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# ----------------------------
# 2) Runtime Stage (실행 전용)
# ----------------------------
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# non-root user 생성
RUN useradd -m appuser

WORKDIR /app

# builder에서 만든 wheel들만 가져와서 설치
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels

# 앱 코드 복사
COPY . .

# 권한 설정
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# FastAPI 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]