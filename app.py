import streamlit as st
import requests
from PIL import Image, ImageDraw
import io
import time
import base64

# Hugging Face API details
API_URL = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"
HF_TOKEN = "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"  # <-- Replace with your token

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# Function to query Hugging Face API
def query_huggingface_api(image_bytes):
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    payload = {
        "inputs": encoded_image
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response

# Function to draw boxes (later if needed)
def draw_boxes(image, detections):
    draw = ImageDraw.Draw(image)
    for detection in detections:
        box = detection['box']
        label = detection['label']
        score = detection['score']
        draw.rectangle([box['xmin'], box['ymin'], box['xmax'], box['ymax']], outline='red', width=3)
        draw.text((box['xmin'], box['ymin'] - 10), f"{label}: {score:.2f}", fill='red')
    return image

# Streamlit app layout
st.title("Object Detection🚀")
st.write("Upload an image and detect objects!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Detect Objects"):
        with st.spinner('Detecting objects... Please wait ⏳'):
            image_bytes = uploaded_file.read()
            max_retries = 5
            for attempt in range(max_retries):
                response = query_huggingface_api(image_bytes)
                
                # 🔥 Debug output
                st.write("Status Code:", response.status_code)
                st.write("Response Content (first 500 bytes):", response.content[:500])

                if response.status_code == 200:
                    try:
                        detections = response.json()
                        st.success("Detection complete!")

                        # Draw detections
                        image_with_boxes = draw_boxes(image.copy(), detections)
                        st.image(image_with_boxes, caption='Detected Objects', use_column_width=True)

                    except Exception as e:
                        st.error(f"Failed to process the detection results: {str(e)}")

                    break
                else:
                    try:
                        error_json = response.json()
                        error_message = error_json.get("error", "Unknown error")
                    except Exception:
                        error_message = "Unknown error"

                    if "loading" in error_message.lower() and attempt < max_retries - 1:
                        st.info(f"Model is loading... retrying ({attempt+1}/{max_retries})...")
                        time.sleep(5)
                    else:
                        st.error(f"Error from Hugging Face API: {error_message}")
                        break
