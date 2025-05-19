import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os

# Constants
MODEL_PATH = "Yolov11best.pt"
class_names = {0: "Awake", 1: "Drowsy"}

# Load model
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

# Check if model exists
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' not found.")
    st.stop()

model = load_model()

# UI setup
st.title("Real-Time Drowsiness Detection")
mode = st.radio("Select Mode", ["Image Upload", "Real-Time Webcam"])

# IMAGE UPLOAD MODE
if mode == "Image Upload":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file).convert("RGB"))
        st.image(image, caption="Input Image", use_container_width=True)

        with st.spinner("Detecting..."):
            results = model.predict(source=image, conf=0.1)
            boxes = results[0].boxes

            if boxes and len(boxes.cls) > 0:
                plotted_img = results[0].plot()
                st.image(plotted_img, caption="Detection Result", use_container_width=True)

                detected_classes = [class_names.get(int(cls), f"Class {int(cls)}") for cls in boxes.cls]
                st.success(f"Detected: {', '.join(detected_classes)}")
            else:
                st.warning("No drowsiness detected.")

# REAL-TIME CAMERA MODE
elif mode == "Real-Time Webcam":
    stframe = st.empty()
    status_text = st.empty()
    start_cam = st.checkbox("Start Camera")

    if start_cam:
        cap = cv2.VideoCapture(0)

        while start_cam:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from webcam.")
                break

            # Resize for faster processing (optional)
            frame_resized = cv2.resize(frame, (640, 640))

            # Run detection
            results = model.predict(source=frame_resized, conf=0.1, verbose=False)
            boxes = results[0].boxes

            # Plot bounding boxes
            plotted_img = results[0].plot()
            plotted_rgb = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)

            # Extract and display predicted labels
            if boxes and len(boxes.cls) > 0:
                detected_classes = [class_names.get(int(cls), f"Class {int(cls)}") for cls in boxes.cls]
                status_text.markdown(f"**Detected:** {', '.join(detected_classes)}")
            else:
                status_text.markdown("**No drowsiness detected.**")

            # Show frame
            stframe.image(plotted_rgb, channels="RGB", use_container_width=True)

        cap.release()
        stframe.empty()
        status_text.empty()
