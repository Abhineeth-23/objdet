import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io

# Page setup
st.set_page_config(page_title="YOLOv8 Detector 🚀", page_icon="🧠", layout="wide")

# Title and description
st.title("🧠 YOLOv8 Object Detection App")
st.caption("Upload an image, set your confidence threshold, and detect objects!")

# Sidebar options
st.sidebar.header("Settings ⚙️")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.01)

# Load model
@st.cache_resource
def load_model():
    model = YOLO('yolov8n.pt')  # nano model
    return model

model = load_model()

# File uploader
uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Button to trigger detection
    if st.button("🚀 Detect Objects"):
        with st.spinner('Running YOLOv8... This may take a few seconds...'):
            img_array = np.array(image)
            results = model.predict(img_array, conf=conf_threshold, iou=iou_threshold)
            result_image = results[0].plot()

            st.image(result_image, caption="🎯 Detection Result", use_column_width=True)
            
            # Save to buffer for download
            buf = io.BytesIO()
            result_pil = Image.fromarray(result_image)
            result_pil.save(buf, format="JPEG")
            byte_im = buf.getvalue()

            st.download_button(
                label="📥 Download Detected Image",
                data=byte_im,
                file_name="detection_result.jpg",
                mime="image/jpeg"
            )