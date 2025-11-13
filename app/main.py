# app/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
import psutil
import os

from app.dependencies import init_models
from app.api.title_search_router import router as title_search_router
from app.api.mf_recommend_router import router as mf_recommend_router
from app.api.anime_translator_router import router as anime_translator_router



@asynccontextmanager
async def lifespan(app: FastAPI):
    # [서버 시작 시 1번 실행]
    await init_models()
    yield
    # [서버 종료 시 실행할 것 (optional)]
    print("Server shutting down...")

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "FastAPI 서버 정상 작동 중!"}

@app.get("/hello/{name}")
def read_item(name: str):
    return {"greeting": f"안녕하세요, {name}님!"}


@app.get("/metrics/memory")
def get_memory_usage():
    """
    현재 FastAPI 프로세스 메모리 사용량 조회 (MB 단위)
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    rss_mb = mem_info.rss / (1024 ** 2)   # 실제 사용 메모리
    vms_mb = mem_info.vms / (1024 ** 2)   # 가상 메모리
    return {
        "rss_mb": round(rss_mb, 2),
        "vms_mb": round(vms_mb, 2),
    }

app.include_router(title_search_router)
app.include_router(mf_recommend_router)
app.include_router(anime_translator_router)