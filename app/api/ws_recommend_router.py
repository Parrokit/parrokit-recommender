# app/api/ws_recommend.py

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
from app.services.recommend_flow_service import recommend_from_titles_with_metadata
from starlette.websockets import WebSocketState

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
            # 소켓이 이미 닫혔거나 닫히는 중이면 전송하지 않음
            if websocket.client_state not in (WebSocketState.CONNECTED, WebSocketState.CONNECTING):
                return
            try:
                await websocket.send_json(event)
            except (RuntimeError, WebSocketDisconnect) as e:
                # close 이후에 늦게 호출된 send는 조용히 무시
                print(f"[ws_recommend] send after close ignored: {e}")
                return

        def progress_cb(event: dict):
            loop.create_task(async_emit(event))

        result = recommend_from_titles_with_metadata(
            titles=titles,
            top_k=top_k,
            cutoff=cutoff,
            exclude_watched=exclude_watched,
            progress_cb=progress_cb,
        )

        # 최종 결과를 클라이언트에 전달 (Flutter 쪽에서 "done" 이벤트를 기다림)
        await async_emit({"event": "done", "result": result})

    except WebSocketDisconnect:
        print("[ws_recommend] client disconnected")
    except Exception as e:
        try:
            if websocket.client_state in (WebSocketState.CONNECTING, WebSocketState.CONNECTED):
                await websocket.send_json({"event": "error", "message": str(e)})
        except (RuntimeError, WebSocketDisconnect):
            # 이미 닫힌 소켓에 대한 에러 전송 시도는 무시
            pass
        finally:
            if websocket.client_state in (WebSocketState.CONNECTING, WebSocketState.CONNECTED):
                await websocket.close()