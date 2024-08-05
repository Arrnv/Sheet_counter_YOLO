import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load the pre-trained YOLOv8 model
model = YOLO('best.pt')  # Replace with the path to your model

# Streamlit app
st.title("YOLOv8 Object Detection App")
st.write("Upload an image to detect objects using YOLOv8.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Resize the image to 640x640 for YOLOv8
    resized_image = cv2.resize(image, (640, 640))

    # Display the uploaded image
    st.image(resized_image, caption='Uploaded Image', use_column_width=True)
    st.write("Processing...")
    average_thickness = 12.5
    # Make predictions with the model
    results = model.predict(resized_image, imgsz=640)

    # Get the class names (you can also load this from your dataset)
    class_names = model.names
    boxes = results[0].boxes.xywh.cpu()
    for box in boxes:
        x, y, w, h = box
        
    no_sheets = int(h)/average_thickness
    # Draw bounding boxes on the image
    annotated_image = resized_image.copy()
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf.item()
            class_idx = int(box.cls.item())
            label = f"{class_names[class_idx]} {confidence:.2f}"  # Use class index to get the name
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            

    # Convert the image to RGB (OpenCV uses BGR by default)
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Display the annotated image
    st.image(annotated_image_rgb, caption='Annotated Image', use_column_width=True)
    st.write(f"No of sheets = {int(no_sheets)}")
    st.write("Done!")
