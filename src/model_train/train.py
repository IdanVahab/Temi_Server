# Importing YOLO model from Ultralytics package
from ultralytics import YOLO  

# Importing Path from pathlib to handle directory paths
from pathlib import Path  

# Importing os to handle file system operations like creating directories
import os  

# Importing random for setting seeds (for reproducibility)
import random  

# Importing torch (PyTorch) for deep learning model handling and GPU use
import torch  

# =======================
# CONFIGURATION SECTION
# =======================

# Path to the dataset configuration YAML file (defines classes, images, annotations)
data_yaml = "C:/Users/Idan Vahab/Desktop/TemiSafetyApp/roboflow/data.yaml"  

# Name of the project folder for saving results
project_name = "yolo_custom_training"  

# Name for this specific training run (used in result folder naming)
run_name = "yolov8s_run"  

# Path to the pretrained YOLO model weights (from previous training)
model_arch = "C:/Users/Idan Vahab/Desktop/TemiSafetyApp/src/yolo_custom_training/yolov8s_run/weights/best.pt"  

# Input image size (640x640 pixels)
img_size = 640  

# Number of training epochs (full passes over the dataset)
epochs = 50  

# Batch size (number of images processed together before weight update)
batch = 8  

# Random seed for reproducibility (ensures results can be replicated)
seed = 42  

# =======================
# OUTPUT & FOLDER SETUP
# =======================

# Define the output directory where results will be saved
output_dir = Path(f"runs/detect/{run_name}")  

# Create the output directory if it doesn’t exist
os.makedirs(output_dir, exist_ok=True)  

# =======================
# REPRODUCIBILITY SETUP
# =======================

# Set Python random seed
random.seed(seed)  

# Set PyTorch random seed for CPU
torch.manual_seed(seed)  

# Set PyTorch random seed for all GPUs
torch.cuda.manual_seed_all(seed)  

# =======================
# LOAD MODEL + START TRAINING
# =======================

# Load YOLO model with specified weights
model = YOLO(model_arch)  

# Run only if this script is the main execution (not imported as a module)
if __name__ == "__main__":  
    # Start training with the defined parameters
    model.train(
        data=data_yaml,             # Dataset YAML path
        epochs=epochs,              # Number of epochs
        imgsz=img_size,             # Input image size
        batch=batch,                # Batch size
        seed=seed,                  # Random seed
        project=project_name,       # Project folder name
        name=run_name,              # Run folder name
        workers=4,                  # Number of CPU workers (for data loading)
        device=0,                   # Device to use (0 = first GPU)
        val=True,                   # Run validation during training
        verbose=True,               # Show detailed training logs
        save=True,                  # Save checkpoints and results
        save_period=1,              # Save after every epoch
        exist_ok=True,              # Allow overwriting previous runs
        pretrained=False,           # Do NOT start from pretrained COCO weights (use our weights)
        resume=False,               # Do NOT resume from last checkpoint
        lr0=0.01,                   # Initial learning rate
        warmup_epochs=3,            # Number of warmup epochs
        patience=10,                # Early stopping patience (stop if no improvement)
        deterministic=True,         # Force deterministic training for reproducibility
        close_mosaic=5,             # Stop using mosaic augmentation after 5 epochs
        auto_augment='randaugment', # Use RandAugment for data augmentation
        iou=0.5,                    # IoU threshold (used in loss calculation)
        conf=0.15,                  # Confidence threshold for object predictions
        box=5.0,                    # Box loss weight
        cls=0.3,                    # Classification loss weight
        dfl=1.5,                    # Distribution focal loss weight
        cos_lr=True,                # Use cosine learning rate scheduler
        dropout=0.1,                # Dropout rate for regularization
        save_conf=True,             # Save confidence scores in outputs
        plots=True                  # Generate training plots
    )

# =======================
# RESULT SUMMARY PRINT
# =======================

# Print completion message with result folder path
print(f"✅ Training complete. Results saved in: {output_dir}")  

# Print instruction to open the metrics image
print(f"📈 To visualize metrics: open {output_dir}/results.png")  

# Print instruction to open the CSV file with detailed results
print(f"📉 To analyze CSV: open {output_dir}/results.csv")  
