# app/api/mf_recommend_router.py
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from app.services.mf_recommend_infer_service import recommend_anime_from_ids
from app.services.mf_recommend_train_service import train_mf_model

router = APIRouter(
    prefix="/mf",
    tags=["mf-recommend"],
)


class MFRecommendRequest(BaseModel):
    anime_ids: List[int] = Field(..., description="사용자가 본(또는 좋아하는) anime_id 리스트")
    top_k: int = Field(20, ge=1, le=100, description="추천 개수")
    exclude_watched: bool = Field(True, description="입력한 애니들은 추천 결과에서 제외할지")


class MFRecommendResponse(BaseModel):
    input_anime_ids: List[int]
    recommended_anime_ids: List[int]


@router.post("/recommend", response_model=MFRecommendResponse)
def mf_recommend(req: MFRecommendRequest):
    """
    {
        "anime_ids": [5114, 9253, 28977],
        "top_k": 10,
        "exclude_watched": true
    }
    """
    try:
        rec_ids = recommend_anime_from_ids(
            liked_anime_ids=req.anime_ids,
            top_k=req.top_k,
            exclude_watched=req.exclude_watched,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return MFRecommendResponse(
        input_anime_ids=req.anime_ids,
        recommended_anime_ids=rec_ids,
    )



class MFTrainRequest(BaseModel):
    epochs: int = Field(10, ge=1, le=100)
    top_k_users: int = Field(500, ge=10, le=10000)
    factors: int = Field(64, ge=4, le=512)
    batch_size: int = Field(4096, ge=32, le=65536)
    csv_path: Optional[str] = "app/data/animelist-dataset/users-score-2023.csv"


@router.post("/train")
def train_mf(req: MFTrainRequest):
    """
    MF 모델 학습 실행 엔드포인트.
    (주의: 시간이 꽤 걸릴 수 있음. 실서비스에서는 비동기/백그라운드 작업 고려해야 함)
    {
        "epochs": 5,
        "top_k_users": 300,
        "factors": 64,
        "batch_size": 4096
    }
    """
    result = train_mf_model(
        csv_path=req.csv_path,
        top_k_users=req.top_k_users,
        factors=req.factors,
        epochs=req.epochs,
        batch_size=req.batch_size,
    )
    return result