import streamlit as st
import requests
from PIL import Image
import io

# Hugging Face API
API_URL = "https://api-inference.huggingface.co/models/Ultralytics/YOLOv8"
HF_TOKEN = "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"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_huggingface_api(image_bytes):
    response = requests.post(API_URL, headers=headers, data=image_bytes)
    return response.content

st.title("YOLOv8 via Hugging Face API 🚀")
st.write("Upload an image and detect objects using Hugging Face API!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Detect Objects"):
        image_bytes = uploaded_file.read()
        output_bytes = query_huggingface_api(image_bytes)

        # Show output image
        output_image = Image.open(io.BytesIO(output_bytes))
        st.image(output_image, caption='Detected Objects', use_column_width=True)
