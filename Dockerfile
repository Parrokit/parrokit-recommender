# Dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 필수 패키지 최소 설치 (필요시만)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# non-root user 생성
RUN useradd -m appuser

WORKDIR /app

# requirements 먼저 복사 (캐시 최적화)
COPY requirements.txt .

# pip 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 앱 전체 복사
COPY . .

# 권한 설정
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# FastAPI 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]