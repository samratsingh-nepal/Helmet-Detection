import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load the YOLOv8 model (pretrained on COCO dataset)
model = YOLO('best.pt')  # You can replace this with your own model

# Streamlit App Title
st.title("YOLOv8 Object Detection with Streamlit")

# Upload image
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Function to run YOLOv8 inference and return image with boxes
def detect_objects(image):
    # Convert PIL image to numpy array (OpenCV format)
    img_array = np.array(image)
    
    # Run YOLOv8 inference
    results = model(img_array)

    # Get bounding boxes and labels from the results
    annotated_frame = results[0].plot()  # Draw bounding boxes on the image

    return annotated_frame

if uploaded_image is not None:
    # Display uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run YOLOv8 model on the uploaded image
    st.write("Detecting objects...")
    result_img = detect_objects(image)

    # Display the output with bounding boxes
    st.image(result_img, caption="Detected Objects", use_column_width=True)

# Add a footer
st.markdown("Powered by YOLOv8 and Streamlit")
