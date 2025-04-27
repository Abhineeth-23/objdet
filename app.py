import streamlit as st
import requests
from PIL import Image
import io

# Hugging Face API
API_URL = "https://api-inference.huggingface.co/models/Ultralytics/YOLOv8"
HF_TOKEN = "your_huggingface_token_here"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_huggingface_api(image_bytes):
    response = requests.post(API_URL, headers=headers, data=image_bytes)
    if response.status_code == 200:
        return response.content
    else:
        return None, response.text

st.title("YOLOv8 via Hugging Face API 🚀")
st.write("Upload an image and detect objects using Hugging Face API!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Detect Objects"):
        image_bytes = uploaded_file.read()
        output = query_huggingface_api(image_bytes)

        if output[0] is not None:
            output_bytes = output[0]
            output_image = Image.open(io.BytesIO(output_bytes))
            st.image(output_image, caption='Detected Objects', use_column_width=True)
        else:
            st.error(f"Error from Hugging Face API: {output[1]}")
