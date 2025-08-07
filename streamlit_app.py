import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from easyocr import Reader
import pytesseract
import tempfile
from detect_and_recognize import detect_number_plates, recognize_number_plates

# Load model and OCR reader once
model = YOLO("runs/detect/train/weights/best.pt")
reader = Reader(['en'], gpu=True)

st.title("🔍 Number Plate Detection and Recognition")

# File uploader (images only)
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.close()

    image = cv2.imread(tfile.name)

    # Run detection
    with st.spinner("Detecting number plates..."):
        boxes = detect_number_plates(image, model, display=False)

    # Run recognition
    if boxes:
        with st.spinner("Recognizing number plates..."):
            plates = recognize_number_plates(image, reader, boxes)

        # Draw results
        for box, text in plates:
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, text, (xmin, ymax + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Detected Plates")
        st.success(f"✅ Detected {len(plates)} plate(s).")
    else:
        st.warning("No number plates detected.")

