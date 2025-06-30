"""
train.py - YOLOv8 custom training script for kitchen scenario detection

This script loads a pre-trained YOLOv8 model and retrains it using a Roboflow dataset
that includes labels such as metal_pot, plate, microwave, and cutlery.
Results are saved to a structured output folder and include plots and CSV metrics.

Requirements:
- Python 3.8
- ultralytics (YOLOv8)
- torch

Author: Idan Vahab
"""

from ultralytics import YOLO
from pathlib import Path
import os
import random
import torch

# =======================
# CONFIGURATION SECTION
# =======================

# Relative path to dataset configuration
data_yaml = Path("roboflow/data.yaml")

# Project and run names
project_name = "yolo_custom_training"
run_name = "yolov8s_run"

# Path to YOLO weights
model_arch = Path("src/model_train/yolo_custom_training/yolov8s_run/weights/best.pt")

# Training hyperparameters
img_size = 640
epochs = 50
batch = 8
seed = 42

# =======================
# OUTPUT SETUP
# =======================

output_dir = Path(f"runs/detect/{run_name}")
os.makedirs(output_dir, exist_ok=True)

# =======================
# REPRODUCIBILITY
# =======================

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# =======================
# LOAD MODEL
# =======================

model = YOLO(str(model_arch))

if __name__ == "__main__":
    """
    Start YOLOv8 training with custom configuration.
    """
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        seed=seed,
        project=project_name,
        name=run_name,
        workers=4,
        device=0,
        val=True,
        verbose=True,
        save=True,
        save_period=1,
        exist_ok=True,
        pretrained=False,
        resume=False,
        lr0=0.01,
        warmup_epochs=3,
        patience=10,
        deterministic=True,
        close_mosaic=5,
        auto_augment='randaugment',
        iou=0.5,
        conf=0.15,
        box=5.0,
        cls=0.3,
        dfl=1.5,
        cos_lr=True,
        dropout=0.1,
        save_conf=True,
        plots=True
    )

    print(f"✅ Training complete. Results saved in: {output_dir}")
    print(f"📈 To visualize metrics: open {output_dir}/results.png")
    print(f"📉 To analyze CSV: open {output_dir}/results.csv")
