import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load the pretrained model
model = YOLO('best.pt')  # Make sure the path to the model is correct

st.title("YOLOv8 Helmet Detection with Streamlit")

# Upload image
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

def detect_helmet(image):
    img_array = np.array(image)  # Convert PIL image to numpy array (OpenCV format)
    
    # Run detection model
    results = model(img_array)  
    
    # Use plot method to visualize results
    result_img = results[0].plot()  # This plots the detected bounding boxes on the image
    
    return result_img

if uploaded_image is not None:
    # Open the uploaded image
    image = Image.open(uploaded_image)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Detect objects
    st.write("Detecting objects...")
    result_img = detect_helmet(image)

    # Convert the result (numpy array) back to a PIL image for display
    result_pil_image = Image.fromarray(result_img)
    
    # Display the result image with detections
    st.image(result_pil_image, caption="Detected Objects", use_column_width=True)

st.markdown("Powered by YOLOv8 and Streamlit")
