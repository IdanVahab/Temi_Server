"""
main.py – FastAPI WebSocket Server Entry Point

This module defines the FastAPI app that handles WebSocket connections
for real-time image frame processing from the TEMI robot.

Routes:
- /ws           : Main endpoint for live YOLO + Scenario recognition
- /save_frames  : Optional endpoint for saving frames from the robot

Author: Idan Vahab
"""

from fastapi import FastAPI, WebSocket
from routes.websocket import websocket_endpoint
from utils.frame_saver import save_frame_from_websocket

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ WebSocket endpoint for real-time YOLO + scenario analysis
@app.websocket("/ws")
async def websocket_route(websocket: WebSocket):
    """
    Main WebSocket route.
    Receives image frames, analyzes with YOLO, sends scenario responses.
    """
    await websocket_endpoint(websocket)

# ✅ WebSocket endpoint to save frames manually (optional)
@app.websocket("/save_frames")
async def save_frames_route(websocket: WebSocket):
    """
    WebSocket route to save incoming frames to disk (for dataset collection).
    """
    await save_frame_from_websocket(websocket)
