!pip install cv2
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Ensure you have the correct model file

st.title("🚀 YOLOv8 Object Detection App")

# Upload image
uploaded_file = st.file_uploader("Upload an Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Perform detection
    results = model(image)

    # Draw bounding boxes with class labels
    for r in results:
        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
            label = model.names[int(cls)]  # Get class name

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Put label text above bounding box
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert OpenCV image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Show image in Streamlit
    st.image(image, caption="Detected Objects", use_container_width=True)
