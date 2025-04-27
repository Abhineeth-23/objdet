import streamlit as st
import requests
from PIL import Image
import io
import time
import base64   # <--- Import base64 here!

# Hugging Face API details
API_URL = "https://api-inference.huggingface.co/models/Ultralytics/YOLOv8"
HF_TOKEN = "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"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# ✅ INSERT THE FUNCTION HERE
def query_huggingface_api(image_bytes):
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    payload = {
        "inputs": encoded_image
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response

# Streamlit app layout
st.title("YOLOv8 Object Detection via Hugging Face 🚀")
st.write("Upload an image and detect objects using Hugging Face API!")

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
                # rest of your retry logic...
