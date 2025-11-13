# app/api/anime_translator_router.py

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from app.services.anime_translator_train_service import train_title_translator
from app.services.anime_translator_infer_service import translate_anime_metadata_by_ids

router = APIRouter(prefix="/translator", tags=["translator"])

class TrainRequest(BaseModel):
    num_train_epochs: int = 8
    learning_rate: float = 2e-4

@router.post("/train")
def train(req: TrainRequest):
    info = train_title_translator(
        num_train_epochs=req.num_train_epochs,
        learning_rate=req.learning_rate,
    )
    return info


class AnimeTranslateRequest(BaseModel):
    anime_ids: List[int]

@router.post("/anime-metadata")
def translate_anime_metadata(req: AnimeTranslateRequest):
    """
    요청 예:
    POST /translator/anime-metadata
    body:
    {
      "anime_ids": [5114, 9253, 30276]
    }
    """
    results = translate_anime_metadata_by_ids(req.anime_ids)
    return {"results": results}