# app/api/recommend_flow_router.py

from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.recommend_flow_service import recommend_from_titles_with_metadata

router = APIRouter(
    prefix="/flow",
    tags=["recommend-flow"],
)


class RecommendFromTitlesRequest(BaseModel):
    titles: List[str] = Field(..., min_items=1, description="사용자가 입력한 애니 제목 리스트")
    top_k: int = Field(10, ge=1, le=50)
    cutoff: float = Field(0.55, ge=0.0, le=1.0)
    exclude_watched: bool = Field(True)



@router.post("/titles-to-metadata")
def recommend_titles_to_metadata(req: RecommendFromTitlesRequest):
    """
    통합 플로우 엔드포인트.

    입력:
    {
      "titles": ["귀멸의 칼날", "나루토 질풍전", "단다단"],
      "top_k": 10,
      "cutoff": 0.55,
      "exclude_watched": true
    }
    """
    try:
        payload = recommend_from_titles_with_metadata(
            titles=req.titles,
            top_k=req.top_k,
            cutoff=req.cutoff,
            exclude_watched=req.exclude_watched,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return payload