import streamlit as st
from PIL import Image
import cv2
from ultralytics import YOLO
import os
import tempfile

MODEL_PATH = "Yolov11best.pt"

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file '{MODEL_PATH}' not found.\n"
        "Please upload the model file to the app folder or provide a download URL."
    )
    st.stop()

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

st.title("Drowsiness Detection App 😴")
st.write("Upload an image or video to detect drowsiness.")

# File uploader
uploaded_file = st.file_uploader("Choose an image or video", type=["jpg", "jpeg", "png", "mp4"])

# Class labels dictionary
class_names = {0: "Awake", 1: "Drowsy"}

if uploaded_file is not None:
    file_type = uploaded_file.type

    if "image" in file_type:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Detecting..."):
            results = model.predict(source=image, conf=0.1)
            boxes = results[0].boxes

            if boxes and len(boxes.cls) > 0:
                plotted_img = results[0].plot()
                plotted_rgb = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)
                st.image(plotted_rgb, caption="Detection Result", use_container_width=True)
                detected_classes = [class_names.get(int(cls), f"Class {int(cls)}") for cls in boxes.cls]
                st.success(f"Detected: {', '.join(detected_classes)}")
            else:
                st.warning("No drowsiness detected.")

    elif "video" in file_type:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        st.video(video_path)
        st.info("Running detection on video (this may take a while)...")

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        out_path = os.path.join("output", "predicted_video.mp4")
        os.makedirs("output", exist_ok=True)
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        with st.spinner("Processing video..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(source=frame, conf=0.1, verbose=False)
                plotted_frame = results[0].plot()
                out.write(plotted_frame)

            cap.release()
            out.release()

        st.success("Video processing complete!")
        st.video(out_path)
