import streamlit as st
import requests
from PIL import Image
import io
import time

# Hugging Face API details
API_URL = "https://api-inference.huggingface.co/models/Ultralytics/YOLOv8"
HF_TOKEN = "your_huggingface_token_here"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# Function to query Hugging Face API
def query_huggingface_api(image_bytes):
    response = requests.post(API_URL, headers=headers, data=image_bytes)
    return response

st.title("Object Detection via Hugging Face 🚀")
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
                if response.status_code == 200:
                    output_bytes = response.content
                    output_image = Image.open(io.BytesIO(output_bytes))
                    st.success("Detection complete!")
                    st.image(output_image, caption='Detected Objects', use_column_width=True)
                    break
                else:
                    try:
                        error_json = response.json()
                        error_message = error_json.get("error", "Unknown error")
                    except Exception:
                        error_message = "Unknown error"

                    if "loading" in error_message.lower() and attempt < max_retries - 1:
                        st.info(f"Model is loading... retrying ({attempt+1}/{max_retries})...")
                        time.sleep(5)  # Wait 5 seconds before retrying
                    else:
                        st.error(f"Error from Hugging Face API: {error_message}")
                        break
