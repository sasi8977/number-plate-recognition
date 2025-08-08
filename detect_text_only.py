import easyocr
from ultralytics import YOLO
import cv2
import os
import re
import numpy as np

MODEL_PATH = "runs/detect/train10/weights/best.pt"
SOURCE_FOLDER = "/home/kishore/Downloads"
OUTPUT_TEXT_FILE = "plate_text_results.txt"

model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'])

def clean_plate_text(text):
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(text) >= 6:  # allow anything with 6+ chars
        return text
    return None

def preprocess_plate(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return thresh

def enlarge_crop(crop, scale=2):
    h, w = crop.shape[:2]
    return cv2.resize(crop, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)

with open(OUTPUT_TEXT_FILE, "w") as f:
    results = model.predict(source=SOURCE_FOLDER, save=False, conf=0.5)

    for result in results:
        img_name = os.path.basename(result.path)
        img = cv2.imread(result.path)
        plate_texts = []

        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crop = enlarge_crop(crop, scale=3)  # make it bigger for OCR
            preprocessed_crop = preprocess_plate(crop)

            ocr_results = []
            for attempt in [crop, preprocessed_crop]:
                result_ocr = reader.readtext(attempt)
                if result_ocr:
                    detected_text = " ".join([res[1] for res in result_ocr])
                    cleaned = clean_plate_text(detected_text)
                    if cleaned:
                        ocr_results.append(cleaned)

            if ocr_results:
                plate_texts.extend(ocr_results)

        final_text = max(set(plate_texts), key=plate_texts.count) if plate_texts else "NO_TEXT_FOUND"
        f.write(f"{img_name} → {final_text}\n")
        print(f"{img_name} → {final_text}")

print(f"\n✅ Results saved to {OUTPUT_TEXT_FILE}")
