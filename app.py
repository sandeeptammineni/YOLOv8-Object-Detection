import torch
from ultralytics import YOLO

# Download YOLOv8 model if not available
model_path = "yolov8n.pt"
try:
    model = YOLO(model_path)
except:
    YOLO(model_path).download()
    model = YOLO(model_path)
