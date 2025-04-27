import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import io

# Load YOLOv8 model
@st.cache_resource
def load_model():
    model = YOLO('yolov8n.pt')  # 'n' = nano (smallest, fastest model)
    return model

model = load_model()

# Function to draw bounding boxes
def draw_boxes(image, detections):
    draw = ImageDraw.Draw(image)

    for det in detections:
        box = det.boxes.xyxy[0].cpu().numpy()  # Get box coordinates
        label = model.names[int(det.boxes.cls[0])]  # Class label
        score = det.boxes.conf[0].cpu().numpy()  # Confidence score

        # Draw rectangle
        draw.rectangle(box, outline="red", width=3)
        # Draw label
        draw.text((box[0], box[1] - 10), f"{label}: {score:.2f}", fill="red")
    
    return image

# Streamlit UI
st.title("YOLOv8 Object Detection (Running Locally 🚀)")
st.write("Upload an image and detect objects using YOLOv8 locally!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Detect Objects"):
        with st.spinner('Detecting objects... Please wait ⏳'):
            # Predict
            results = model.predict(np.array(image))

            # Draw detections
            output_image = draw_boxes(image.copy(), results)

            st.success("Detection complete!")
            st.image(output_image, caption='Detected Objects', use_column_width=True)
