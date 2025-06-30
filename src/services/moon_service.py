import asyncio
from utils.moon_model import describe_image_with_moondream, get_moondream_question
from fastapi import WebSocket
import numpy as np

# Global flag to prevent concurrent MoonDream requests
is_moondream_busy = False

async def send_moondream_result(websocket: WebSocket, image: np.ndarray, labels: set):
    """
    Asynchronously sends an image and detected labels to MoonDream for analysis.
    Prevents concurrent processing via a global lock.

    Args:
        websocket (WebSocket): WebSocket connection to send result back to client.
        image (np.ndarray): Image frame (decoded from base64).
        labels (set): Set of normalized labels detected in the image.

    Sends:
        JSON response via WebSocket containing:
            - source: "moondream"
            - question: auto-generated question about the image
            - answer: visual description from MoonDream model
    """
    global is_moondream_busy

    if is_moondream_busy:
        print("⏳ Skipping MoonDream – still processing previous request")
        return

    try:
        is_moondream_busy = True

        # Run MoonDream analysis in a thread-safe way
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
