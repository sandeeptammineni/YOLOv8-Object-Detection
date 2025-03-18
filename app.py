import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

# Download YOLOv8 model if not available
model_path = "yolov8n.pt"
if not os.path.exists(model_path):
    st.write("Downloading YOLOv8 model...")
    model = YOLO("yolov8n.pt")
else:
    model = YOLO(model_path)

st.title("🚀 YOLOv8 Object Detection App")

# Upload image
uploaded_file = st.file_uploader("Upload an Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Perform detection
    results = model(image)

    # Draw bounding boxes
    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Convert OpenCV image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Show image in Streamlit
    st.image(image, caption="Detected Objects", use_column_width=True)
