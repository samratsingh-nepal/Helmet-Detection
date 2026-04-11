import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

# Load model with error handling
model_path = 'best.pt'
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please upload or provide the model.")
    st.stop()

try:
    model = YOLO(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.title("YOLOv8 Helmet Detection with Streamlit")

uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

def detect_helmet(image):
    # Convert PIL image to numpy array (RGB)
    img_array = np.array(image)
    # Run inference
    results = model(img_array)
    # Plot bounding boxes on the image (returns BGR numpy array)
    result_img_bgr = results[0].plot()
    # Convert BGR to RGB for correct display in Streamlit
    result_img_rgb = cv2.cvtColor(result_img_bgr, cv2.COLOR_BGR2RGB)
    return result_img_rgb

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Detecting helmets..."):
        result_img = detect_helmet(image)

    st.image(result_img, caption="Detected Objects", use_column_width=True)

st.markdown("Powered by YOLOv8 and Streamlit")
