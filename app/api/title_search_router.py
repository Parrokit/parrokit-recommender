# app/api/title_search_router.py
from typing import List

from fastapi import APIRouter, Query
from app.services.title_search_infer_service import search_anime_titles, batch_search_anime_titles

router = APIRouter(
    prefix="/titles",
    tags=["titles"],
)


@router.get("/search")
def search_title_api(
    query: str = Query(..., description="검색할 애니 제목 (한글/영문/일문 아무거나)"),
    k: int = Query(5, ge=1, le=20, description="상위 몇 개까지 가져올지"),
    cutoff: float = Query(0.55, ge=0.0, le=1.0, description="최소 코사인 유사도 임계값"),
):
    """
    단일 질의로 애니 제목 검색 API.
    예: GET /titles/search?query=귀멸의+칼날&k=5
    """
    results = search_anime_titles(query=query, k=k, cutoff=cutoff)
    return {
        "query": query,
        "k": k,
        "cutoff": cutoff,
        "results": results,
    }


@router.post("/batch-search")
def batch_search_title_api(
    queries: List[str],
    k: int = 5,
    cutoff: float = 0.55,
):
    """
    여러 질의를 한 번에 검색하는 API.
    Body 예시:
    {
      "queries": ["귀멸의 칼날", "나루토 질풍전", "단다단"]
    }
    """
    results = batch_search_anime_titles(queries=queries, k=k, cutoff=cutoff)
    return {
        "queries": queries,
        "k": k,
        "cutoff": cutoff,
        "results": results,
    }