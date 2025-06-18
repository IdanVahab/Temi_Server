from fastapi import FastAPI, WebSocket
from routes.websocket import websocket_endpoint
from utils.frame_saver import save_frame_from_websocket

app = FastAPI()

@app.websocket("/ws")
async def websocket_route(websocket: WebSocket):
    await websocket_endpoint(websocket)

@app.websocket("/save_frames")
async def save_frames_route(websocket: WebSocket):
    await save_frame_from_websocket(websocket)
