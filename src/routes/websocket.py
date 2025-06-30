from fastapi import WebSocket, WebSocketDisconnect
import time, asyncio
from services.yolo_service import process_frame_and_predict
from services.moon_service import send_moondream_result
from utils.scenario_handler import ScenarioHandler

# List to keep track of connected clients
connected_clients = []

# Timestamp of the last time MoonDream was triggered
last_moondream_sent = 0

# Interval in seconds between MoonDream triggers
MOONDREAM_INTERVAL = 3.0

async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket handler for receiving frames from the TEMI robot.

    Responsibilities:
    - Accept incoming WebSocket connections from the robot.
    - Receive base64-encoded image frames continuously.
    - Run object detection (YOLO) and object tracking (Deep SORT).
    - Update the scenario handler with label history and tracked objects.
    - Detect high-level scenarios and send them back to the robot.
    - Optionally trigger MoonDream analysis every few seconds.

    Args:
        websocket (WebSocket): WebSocket connection with the TEMI robot client.
    """
    global last_moondream_sent

    await websocket.accept()
    connected_clients.append(websocket)
    print("📡 Client connected")

    # Initialize the scenario handler for this session
    scenario_handler = ScenarioHandler()
    last_labels_sent_to_moondream = set()

    try:
        while True:
            # Step 1: Receive image frame (base64 string)
            data = await websocket.receive_text()
            print("🖼 Received new frame from client")

            # Step 2: Process image via YOLO + Deep SORT
            image, prediction, normalized_labels, tracked_objects = await process_frame_and_predict(data)
            print(f"🔎 YOLO Prediction: {prediction}")
            print(f"🏷️ Labels: {normalized_labels}")
            print(f"🎯 Tracked Objects: {len(tracked_objects)}")

            # Step 3: Send back raw prediction if exists
            if prediction:
                await websocket.send_text(prediction)

            # Step 4: Update scenario logic
            scenario_handler.update(normalized_labels)
            scenario_handler.update_tracking(tracked_objects)
            scenario = scenario_handler.get_active_scenario()

            # Step 5: If scenario detected, send it to the robot
            if scenario:
                scenario_name = scenario['scenario']
                # incident_id = scenario['incident_id']
                message = scenario_name  # or include incident if desired
                print(f"⚠️ Scenario Detected: {message}")
                await websocket.send_text(message)

            # Step 6: Optional MoonDream analysis
            if (
                time.time() - last_moondream_sent >= MOONDREAM_INTERVAL and
                normalized_labels and
                normalized_labels != last_labels_sent_to_moondream
            ):
                last_moondream_sent = time.time()
                last_labels_sent_to_moondream = normalized_labels
                print("🧠 Sending to MoonDream")
                asyncio.create_task(send_moondream_result(websocket, image, normalized_labels))

    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        print("❌ Client disconnected")
