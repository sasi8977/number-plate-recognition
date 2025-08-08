import easyocr
from ultralytics import YOLO
import cv2
import os
import re
import numpy as np

# ===== SETTINGS =====
MODEL_PATH = "runs/detect/train10/weights/best.pt"  # Path to trained YOLO model
SOURCE_FOLDER = "/home/kishore/Downloads"           # Folder with test images
OUTPUT_TEXT_FILE = "plate_text_results.txt"         # File to save plate numbers
# ====================

# Load YOLO model
model = YOLO(MODEL_PATH)

# Load EasyOCR reader
reader = easyocr.Reader(['en'])

# Function to clean OCR result for Indian plates
def clean_plate_text(text):
    text = re.sub(r'[^A-Z0-9]', '', text.upper())  # Remove unwanted chars
    match = re.match(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{1,4}$', text)
    return match.group(0) if match else None

# Preprocess image for OCR
def preprocess_plate(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Reduce noise
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    return thresh

# Open text file for writing results
with open(OUTPUT_TEXT_FILE, "w") as f:
    results = model.predict(source=SOURCE_FOLDER, save=False, conf=0.5)

    for result in results:
        img_name = os.path.basename(result.path)
        img = cv2.imread(result.path)
        plate_texts = []

        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            crop = img[y1:y2, x1:x2]
            preprocessed_crop = preprocess_plate(crop)

            ocr_result = reader.readtext(preprocessed_crop)
            if ocr_result:
                detected_text = " ".join([res[1] for res in ocr_result])
                cleaned = clean_plate_text(detected_text)
                if cleaned:
                    plate_texts.append(cleaned)

        if plate_texts:
            final_text = max(set(plate_texts), key=plate_texts.count)  # Most common
        else:
            final_text = "NO_TEXT_FOUND"

        f.write(f"{img_name} → {final_text}\n")
        print(f"{img_name} → {final_text}")

print(f"\n✅ Results saved to {OUTPUT_TEXT_FILE}")
