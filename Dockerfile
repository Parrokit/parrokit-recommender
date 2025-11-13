# Dockerfile (conda-based)
FROM continuumio/miniconda3:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create non-root user
RUN useradd -m appuser

WORKDIR /app

# 1) env 만들 때 필요한 파일들 먼저 복사
COPY environment.yml .
COPY requirements.txt .

# 2) conda env 생성
RUN conda env create -f environment.yml

# 3) 이후부터는 parrokit env 기준으로 실행
SHELL ["conda", "run", "-n", "parrokit", "/bin/bash", "-c"]

# 4) 애플리케이션 전체 복사
COPY . .

# 5) 권한 설정 + 비루트 유저
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 6) FastAPI 실행
CMD ["conda", "run", "-n", "parrokit", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]