
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import io

st.set_page_config(page_title="Object Detection App 🚀", layout="wide")

st.title("🔍 Object Detection with YOLOv8")
st.write("Upload an image and detect objects easily!")

confidence_threshold = st.slider('Confidence Threshold', 0.0, 1.0, 0.25, 0.01)

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Objects"):
        with st.spinner('Running Detection...'):
            model = YOLO('yolov8n.pt')  # Using nano model for faster inference
            results = model.predict(image, conf=confidence_threshold)

            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            scores = results[0].boxes.conf.cpu().numpy()

            draw = ImageDraw.Draw(image)
            for box, cls, score in zip(boxes, classes, scores):
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.text((x1, y1), f"{model.names[cls]} {score:.2f}", fill="red")

            st.success("Detection Complete!")
            st.image(image, caption="Detected Objects", use_column_width=True)

            # Save to BytesIO
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)

            st.download_button(
                label="Download Result Image 🎯",
                data=img_bytes,
                file_name="detection_result.png",
                mime="image/png"
            )
