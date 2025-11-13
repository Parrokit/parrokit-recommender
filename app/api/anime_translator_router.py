# app/api/anime_translator_router.py

from fastapi import APIRouter
from pydantic import BaseModel
from app.services.anime_translator_train_service import train_title_translator

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