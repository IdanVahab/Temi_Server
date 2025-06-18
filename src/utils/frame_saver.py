import os
import io
import base64
import time
from PIL import Image
from fastapi import WebSocket
from datetime import datetime

# הגדר תיקייה לשמירת תמונות
SAVE_DIR = "saved_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

# שמור כמה זמן עבר מהשמירה האחרונה
last_saved_time = 0
SAVE_INTERVAL = 0.1  # שניות (כל 0.1 שנייה)

async def save_frame_from_websocket(websocket: WebSocket):
    global last_saved_time
    await websocket.accept()
    print("🖼️ Frame saving started...")

    try:
        while True:
            data = await websocket.receive_text()
            now = time.time()

            if now - last_saved_time >= SAVE_INTERVAL:
                last_saved_time = now

                # Decode and save image
                image_data = base64.b64decode(data)
                image = Image.open(io.BytesIO(image_data)).convert("RGB")

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{SAVE_DIR}/frame_{timestamp}.jpg"
                image.save(filename)
                print(f"✅ Saved: {filename}")

    except Exception as e:
        print(f"❌ Error saving frame: {e}")
