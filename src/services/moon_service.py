import asyncio
from utils.moon_model import describe_image_with_moondream, get_moondream_question
from fastapi import WebSocket
import numpy as np

is_moondream_busy = False

is_moondream_busy = False

async def send_moondream_result(websocket: WebSocket, image: np.ndarray, labels: set):
    global is_moondream_busy
    if is_moondream_busy:
        print("⏳ Skipping MoonDream – still processing previous request")
        return
    try:
        is_moondream_busy = True
        loop = asyncio.get_event_loop()
        description = await loop.run_in_executor(None, describe_image_with_moondream, image, labels)
        question = get_moondream_question(labels)

        print(f"🧠 MoonDream analysis: {description}")
        await websocket.send_json({
            "source": "moondream",
            "question": question,
            "answer": description
        })
    except Exception as e:
        print(f"❌ Error in send_moondream_result: {e}")
    finally:
        is_moondream_busy = False

