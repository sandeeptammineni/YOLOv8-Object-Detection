import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# ✅ Load YOLOv8 model
model = YOLO("yolov8n.pt")  

st.title("🚀 YOLOv8 Object Detection App")

# ✅ Upload image
uploaded_file = st.file_uploader("Upload an Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # ✅ Convert the uploaded image to OpenCV format
    image = Image.open(uploaded_file)
    image = np.array(image)

    # ✅ Perform object detection
    results = model(image)

    # ✅ Draw bounding boxes with class labels
    for r in results:
        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            x1, y1, x2, y2 = map(int, box)  
            label = model.names[int(cls)]  

            # ✅ Draw bounding box & label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ✅ Convert OpenCV image to RGB format for Streamlit
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ✅ Display detected image in Streamlit
    st.image(image, caption="Detected Objects", use_container_width=True)
