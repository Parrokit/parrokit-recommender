# app/api/ws_recommend.py (예시)

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.recommend_flow_service import recommend_from_titles_with_metadata

router = APIRouter()

@router.websocket("/ws/recommend")
async def ws_recommend(websocket: WebSocket):
    await websocket.accept()

    try:
        # 1) 클라이언트에서 첫 메시지로 titles 같은 걸 보내준다고 가정
        data = await websocket.receive_json()
        titles = data.get("titles", [])
        top_k = int(data.get("top_k", 10))

        # 2) 진행 상황을 WebSocket으로 쏘는 콜백 정의
        async def async_emit(event: dict):
            await websocket.send_json(event)

        # sync 함수 안에서 async 콜백을 쓰려면 run_until_complete 같은 게 필요해서
        # 간단하게는 동기 래퍼를 하나 두는 방법이 있음
        import asyncio
        loop = asyncio.get_event_loop()

        def progress_cb(event: dict):
            # 백그라운드에서 비동기로 전송
            loop.create_task(async_emit(event))

        # 3) 추천 수행
        result = recommend_from_titles_with_metadata(
            titles=titles,
            top_k=top_k,
            progress_cb=progress_cb,
        )

        # result는 이미 "done" 이벤트로 한 번 나가지만,
        # 혹시 몰라서 여기서 한 번 더 최종 결과를 보내고 소켓을 닫을 수도 있음
        # await websocket.send_json({"event": "done", "result": result})

        await websocket.close()

    except WebSocketDisconnect:
        print("[INFO] client disconnected")
    except Exception as e:
        # 에러도 이벤트로 던져줄 수 있음
        try:
            await websocket.send_json({"event": "error", "message": str(e)})
        finally:
            await websocket.close()