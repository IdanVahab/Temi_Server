
from utils.scenario_handler import ScenarioHandler
from services.yolo_service import yolo_model, normalize_class_names, classify_scenario, tracker
import os
import cv2
import numpy as np
from datetime import datetime


# Initialize scenario handler
scenario_handler = ScenarioHandler()

# Directory with saved images
image_dir = "saved_frames"
output_dir = "annotated_output"
os.makedirs(output_dir, exist_ok=True)

results = []

# Go through each image in the folder
for filename in sorted(os.listdir(image_dir)):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(image_dir, filename)
        image = cv2.imread(path)

        # Run YOLO
        results_yolo = yolo_model.predict(image, conf=0.3)
        boxes = results_yolo[0].boxes
        names = results_yolo[0].names
        annotated_image = results_yolo[0].plot()

        if not boxes or boxes.cls is None:
            detected_labels = set()
            tracked_objects = []
        else:
            detected_classes = boxes.cls.cpu().numpy().astype(int)
            class_names = [names[i] for i in detected_classes]
            normalized = normalize_class_names(class_names)
            detected_labels = set(normalized)

            # Format for Deep SORT
            detections = []
            for box, cls, conf in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = box
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, names[int(cls)]))

            tracked_objects = []
            tracks = tracker.update_tracks(detections, frame=image)
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
                # Draw track ID on annotated image
                cv2.putText(annotated_image, f"{label} ({track_id})", (int(l), int(t) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Update scenario handler
        scenario_handler.update(detected_labels)
        scenario_handler.update_tracking(tracked_objects)
        scenario = scenario_handler.get_active_scenario()

        # Save annotated image
        out_path = os.path.join(output_dir, f"annotated_{filename}")
        cv2.imwrite(out_path, annotated_image)

        print(f"{filename} | Labels: {list(detected_labels)} | Scenario: {scenario} | Saved: {out_path}")