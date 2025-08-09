import cv2
import pytesseract
import easyocr
import re
from pathlib import Path
from ultralytics import YOLO
import os
from datetime import datetime

# ---------------- CONFIG ---------------- #
MODEL_PATH = "best.pt"  # Your trained YOLO model
SOURCE_DIR = "/home/kishore/Downloads"  # Folder with test images
RESULTS_FILE = "plate_text_results.txt"
DEBUG_CROPS_DIR = "debug_crops"
DEBUG_PROCESSED_DIR = "debug_processed"
PADDING = 15  # Extra pixels around YOLO detection box
# ----------------------------------------- #

# Prepare output folders
os.makedirs(DEBUG_CROPS_DIR, exist_ok=True)
os.makedirs(DEBUG_PROCESSED_DIR, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)

# Load EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# Regex for Indian number plates
PLATE_REGEX = r'[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}'

# Tesseract configs
STRICT_OCR = '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --dpi 300'
LOOSE_OCR = '--psm 6 --oem 3 --dpi 300'

# Preprocessing function
def preprocess_plate(crop, aggressive=True):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    if aggressive:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.bilateralFilter(gray, 9, 15, 15)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    else:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh

# Get all images
image_paths = [p for p in Path(SOURCE_DIR).glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
if not image_paths:
    print(f"❌ No images found in {SOURCE_DIR}")
    exit()

results_text = []

for img_path in image_paths:
    mod_time = datetime.fromtimestamp(img_path.stat().st_mtime)
    print(f"\n📂 Processing: {img_path} (Last Modified: {mod_time})")

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    detections = model(img)[0]
    best_guess = "UNREADABLE"
    engine_used = "None"

    for i, box in enumerate(sorted(detections.boxes, key=lambda b: b.conf[0], reverse=True)):
        if box.conf[0] < 0.5:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Apply padding
        x1 = max(0, x1 - PADDING)
        y1 = max(0, y1 - PADDING)
        x2 = min(img.shape[1], x2 + PADDING)
        y2 = min(img.shape[0], y2 + PADDING)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Save crop
        crop_path = f"{DEBUG_CROPS_DIR}/{img_path.stem}_plate{i+1}.jpg"
        cv2.imwrite(crop_path, crop)

        # Pass 1: Aggressive preprocessing + Tesseract strict
        processed = preprocess_plate(crop, aggressive=True)
        cv2.imwrite(f"{DEBUG_PROCESSED_DIR}/{img_path.stem}_plate{i+1}_proc.jpg", processed)
        raw_strict = pytesseract.image_to_string(processed, config=STRICT_OCR).strip().replace(" ", "").upper()
        match_strict = re.search(PLATE_REGEX, raw_strict)

        # Pass 2: Aggressive preprocessing + Tesseract loose
        raw_loose = pytesseract.image_to_string(processed, config=LOOSE_OCR).strip().replace(" ", "").upper()
        match_loose = re.search(PLATE_REGEX, raw_loose)

        # Pass 3: EasyOCR fallback if no match yet
        if not (match_strict or match_loose):
            easy_texts = reader.readtext(crop, detail=0)
            easy_joined = "".join(easy_texts).replace(" ", "").upper()
            match_easy = re.search(PLATE_REGEX, easy_joined)
        else:
            match_easy = None
            easy_joined = ""

        # Decide final
        if match_strict:
            best_guess = match_strict.group(0)
            engine_used = "Tesseract-Strict"
        elif match_loose:
            best_guess = match_loose.group(0)
            engine_used = "Tesseract-Loose"
        elif match_easy:
            best_guess = match_easy.group(0)
            engine_used = "EasyOCR"
        else:
            best_guess = "UNREADABLE"
            engine_used = "None"

        print(f"{img_path.name} → Conf: {box.conf[0]:.2f} → RAW_STRICT: {raw_strict} → RAW_LOOSE: {raw_loose} → EASY: {easy_joined} → FINAL: {best_guess} → Engine: {engine_used}")

        results_text.append(
            f"{img_path.name} → Conf: {box.conf[0]:.2f} → RAW_STRICT: {raw_strict} → RAW_LOOSE: {raw_loose} → EASY: {easy_joined} → FINAL: {best_guess} → Engine: {engine_used}"
        )

# Save results to file
with open(RESULTS_FILE, "w") as f:
    f.write("\n".join(results_text))

print(f"\n✅ Results saved to {RESULTS_FILE}")
print(f"🔍 Cropped plates saved in '{DEBUG_CROPS_DIR}', processed plates in '{DEBUG_PROCESSED_DIR}'")




