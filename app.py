import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image


model = YOLO('best.pt')  # pretrained for helmet detection

st.title("YOLOv8 Helmet Detection with Streamlit")


uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"]) # Upload image

def detect_helmet(image):
    
    results = model(image)   # Run best.pt model
    
    return results

if uploaded_image is not None:
    image = Image.open(uploaded_image)  # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

   
    st.write("Detecting objects...")
    result_img = detect_helmet(image)
    image = Image.open(result_img)  # Display final image
    st.image(image, caption="Final Image", use_column_width=True)
   

st.markdown("Powered by YOLOv8 and Streamlit")
