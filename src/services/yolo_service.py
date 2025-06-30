from ultralytics import YOLO
import base64, io, traceback
import numpy as np
from PIL import Image
import torch
from utils.helpers import normalize_class_names, classify_scenario
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import os
from datetime import datetime
from pathlib import Path

# ✅ Load YOLO model with relative path
base_dir = Path(__file__).resolve().parent
model_path = base_dir / "../model_train/yolo_custom_training/yolov8s_run/weights/best.pt"
model_path = str(model_path.resolve())
yolo_model = YOLO(model_path)
if torch.cuda.is_available():
    yolo_model.to("cuda")

# ✅ Initialize Deep SORT
tracker = DeepSort(max_age=30)

# def save_image(image_np, prefix="frame"):
#     os.makedirs("saved_frames", exist_ok=True)
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
#     path = f"saved_frames/{prefix}_{timestamp}.jpg"
#     cv2.imwrite(path, image_np)
#     print(f"🖼 Frame saved: {path}")

async def process_frame_and_predict(base64_string: str):
    """
    Decodes a base64-encoded image, performs YOLOv8 object detection and Deep SORT tracking,
    then classifies the scenario based on detected objects.

    Args:
        base64_string (str): Base64-encoded image sent from the TEMI robot.

    Returns:
        Tuple:
            - img_np (np.ndarray): Decoded OpenCV image.
            - prediction (str): Classified scenario name (e.g., "pouring_food").
            - normalized (set): Set of normalized class labels detected.
            - tracked_objects (list): List of tracked objects with IDs, labels, and bounding boxes.
    """
    try:
        # ✅ Decode and prepare image
        image_data = base64.b64decode(base64_string)
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        img_np = cv2.imdecode(image_array, cv2.IMREAD_COLOR)  # זה יחזיר BGR עם צבעים נכונים

        # ✅ Run YOLO
        results = yolo_model.predict(img_np, conf=0.3)
        boxes = results[0].boxes
        names = results[0].names

        # ✅ Save image for debugging
        # save_image(img_np, prefix="yolo")

        if not boxes or boxes.cls is None:
            return img_np, "no_objects", set(), []

        detected_classes = boxes.cls.cpu().numpy().astype(int)
        class_names = [names[i] for i in detected_classes]
        normalized = normalize_class_names(class_names)
        prediction = classify_scenario(normalized)

        # ✅ Convert YOLO boxes to format: [x1, y1, x2, y2, confidence, class_id]
        detections = []
        for box, cls, conf in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, names[int(cls)]))

        # ✅ Track with Deep SORT
        tracked_objects = []
        tracks = tracker.update_tracks(detections, frame=img_np)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, w, h = track.to_ltrb()
            label = track.get_det_class() if track.get_det_class() else "object"
            tracked_objects.append({
                "id": track_id,
                "label": label,
                "bbox": [int(l), int(t), int(w), int(h)]
            })

        return img_np, prediction, normalized, tracked_objects

    except Exception as e:
        print(f"❌ YOLO/DeepSORT error: {e}")
        traceback.print_exc()
        return None, None, set(), []
