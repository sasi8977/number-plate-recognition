import os
import re
from pathlib import Path
from ultralytics import YOLO
import cv2
import easyocr
import pytesseract
from pytesseract import Output

# =====================
# USER SETTINGS
# =====================
MODEL_PATH = "/home/kishore/Downloads/number-plate-recognition-main/runs/detect/train10/weights/best.pt"
SOURCE_DIR = "/home/kishore/Downloads/test_images"
RESULTS_FILE = "plate_text_results.txt"
PADDING = 15
PLATE_REGEX = r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{1,4}$'  # Indian plate format
# =====================

# Load YOLO model
model = YOLO(MODEL_PATH)

# Load OCR
easyocr_reader = easyocr.Reader(['en'], gpu=False)

# Ensure source exists
if not os.path.exists(SOURCE_DIR):
    raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR}")

# Function: Validate plate
def is_valid_plate(text):
    return re.match(PLATE_REGEX, text.replace(" ", "")) is not None

# Function: Preprocess for OCR
def preprocess_for_ocr(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# Function: Run OCR
def run_ocr(crop):
    crop_proc = preprocess_for_ocr(crop)

    # Try EasyOCR
    results_easy = easyocr_reader.readtext(crop_proc, detail=0)
    results_easy = [r.upper().replace(" ", "") for r in results_easy if len(r) >= 6]

    # Try Tesseract
    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text_tess = pytesseract.image_to_string(crop_proc, config=config)
    text_tess = text_tess.strip().upper().replace(" ", "")

    candidates = []
    if results_easy:
        candidates.extend(results_easy)
    if text_tess:
        candidates.append(text_tess)

    # Keep only valid plates
    valid = [c for c in candidates if is_valid_plate(c)]
    return valid[0] if valid else ""

# Main
with open(RESULTS_FILE, "w") as f_out:
    for file_name in os.listdir(SOURCE_DIR):
        if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(SOURCE_DIR, file_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[WARN] Cannot read image: {img_path}")
            continue

        results = model.predict(source=img_path, conf=0.5, verbose=False)

        best_text = ""
        best_conf = 0.0

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # Apply padding
                x1 = max(0, x1 - PADDING)
                y1 = max(0, y1 - PADDING)
                x2 = min(img.shape[1], x2 + PADDING)
                y2 = min(img.shape[0], y2 + PADDING)

                crop = img[y1:y2, x1:x2]

                text = run_ocr(crop)

                if text and conf > best_conf:
                    best_text = text
                    best_conf = conf

        line = f"{file_name} | {best_text if best_text else 'NO_PLATE'} | YOLO: {best_conf:.2f}"
        print(line)
        f_out.write(line + "\n")

print(f"\n✅ Detection completed. Results saved to: {RESULTS_FILE}")
