import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image


model = YOLO('best.pt')  # Loading pre-trained model

st.title("YOLOv8 Helmet Detection with Streamlit")


uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"]) # Upload image

def detect_helmet(image):
    img_array = np.array(image)  # Convert PIL image to numpy array (OpenCV format)
    
   
    results = model(img_array)   # Run detection model
    
    
    result_img = results[0].plot()  # detects bounding boxes on the image
    
    return result_img

if uploaded_image is not None:
   
    image = Image.open(uploaded_image)  # Open the uploaded image
    
  
    st.image(image, caption="Uploaded Image", use_column_width=True)   # Display the uploaded image

   
    st.write("Detecting Helmets...")  # Detect Helmet
    result_img = detect_helmet(image)

    
    result_pil_image = Image.fromarray(result_img) # Convert numpy array back to a PIL image for display
    
    
    st.image(result_pil_image, caption="Detected Objects", use_column_width=True) # Display the result image with detections

st.markdown("Powered by YOLOv8 and Streamlit")
