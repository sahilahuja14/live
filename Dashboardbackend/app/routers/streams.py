from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..services.stream_manager import stream_manager

router = APIRouter()

@router.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    await stream_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, maybe wait for client messages if needed
            # For now, it's a one-way stream from server to client
            # Keep connection alive, listen for client config messages
            data = await websocket.receive_text()
            await stream_manager.handle_client_message(websocket, data) 
    except WebSocketDisconnect:
        stream_manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        stream_manager.disconnect(websocket)
