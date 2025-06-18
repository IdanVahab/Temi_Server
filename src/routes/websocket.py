from fastapi import WebSocket, WebSocketDisconnect
import time, asyncio
from services.yolo_service import process_frame_and_predict
from services.moon_service import send_moondream_result
from utils.scenario_handler import ScenarioHandler

connected_clients = []
last_moondream_sent = 0
MOONDREAM_INTERVAL = 3.0

async def websocket_endpoint(websocket: WebSocket):
    global last_moondream_sent
    await websocket.accept()
    connected_clients.append(websocket)
    print("📡 Client connected")

    scenario_handler = ScenarioHandler()
    last_labels_sent_to_moondream = set()

    try:
        while True:
            data = await websocket.receive_text()
            print("🖼 Received new frame from client")

            image, prediction, normalized_labels, tracked_objects = await process_frame_and_predict(data)

            print(f"🔎 YOLO Prediction: {prediction}")
            print(f"🏷️ Labels: {normalized_labels}")
            print(f"🎯 Tracked Objects: {len(tracked_objects)}")

            if prediction:
                await websocket.send_text(prediction)

            scenario_handler.update(normalized_labels)
            scenario_handler.update_tracking(tracked_objects)
            scenario = scenario_handler.get_active_scenario()
            if scenario:
                scenario_name = scenario['scenario']
                # incident_id = scenario['incident_id']

                # if incident_id:
                #     message = f"⚠️ {scenario_name} (incident: {incident_id})"
                # else:
                message = scenario_name

                print(f"⚠️ Scenario Detected: {message}")
                await websocket.send_text(message)

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
