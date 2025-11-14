# app/api/ws_recommend.py

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
from app.services.recommend_flow_service import recommend_from_titles_with_metadata

router = APIRouter()

@router.websocket("/ws/recommend")
async def ws_recommend(websocket: WebSocket):
    await websocket.accept()

    try:
        data = await websocket.receive_json()
        titles = data.get("titles", [])
        top_k = int(data.get("top_k", 10))
        cutoff = float(data.get("cutoff", 0.55))
        exclude_watched = bool(data.get("exclude_watched", True))

        loop = asyncio.get_event_loop()

        async def async_emit(event: dict):
            await websocket.send_json(event)

        def progress_cb(event: dict):
            loop.create_task(async_emit(event))

        result = recommend_from_titles_with_metadata(
            titles=titles,
            top_k=top_k,
            cutoff=cutoff,
            exclude_watched=exclude_watched,
            progress_cb=progress_cb,
        )

        # 필요하면 여기서도 한 번 더 최종 결과를 보내도 됨
        # await websocket.send_json({"event": "done", "result": result})

        await websocket.close()

    except WebSocketDisconnect:
        print("[ws_recommend] client disconnected")
    except Exception as e:
        try:
            await websocket.send_json({"event": "error", "message": str(e)})
        finally:
            await websocket.close()