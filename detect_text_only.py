import cv2
import pytesseract
import re
from pathlib import Path
from ultralytics import YOLO
import os
from datetime import datetime

# ---------------- CONFIG ---------------- #
MODEL_PATH = "runs/detect/train10/weights/best.pt"  # Update if needed
SOURCE_DIR = "/home/kishore/Downloads"              # Folder with test images
RESULTS_FILE = "plate_text_results.txt"
DEBUG_CROPS_DIR = "debug_crops"
DEBUG_PROCESSED_DIR = "debug_processed"
# ----------------------------------------- #

# Make debug dirs
os.makedirs(DEBUG_CROPS_DIR, exist_ok=True)
os.makedirs(DEBUG_PROCESSED_DIR, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)

# Regex for Indian number plates
PLATE_REGEX = r'[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}'

# OCR configs
STRICT_OCR = '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --dpi 300'
LOOSE_OCR = '--psm 6 --oem 3 --dpi 300'

# Preprocessing
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
        thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
    
    return thresh

# Get all images
image_paths = [
    p for p in Path(SOURCE_DIR).glob("*")
    if p.suffix.lower() in (".jpg", ".jpeg", ".png")
]

if not image_paths:
    print(f"❌ No images found in {SOURCE_DIR}")
    exit()

results_text = []

for img_path in sorted(image_paths, key=lambda x: x.stat().st_mtime, reverse=True):
    print(f"\n📂 Processing: {img_path} (Last Modified: {datetime.fromtimestamp(img_path.stat().st_mtime)})")

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    detections = model(img)[0]
    best_guess = "UNREADABLE"
    raw_best = ""

    for i, box in enumerate(sorted(detections.boxes, key=lambda b: b.conf[0], reverse=True)):
        if box.conf[0] < 0.5:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Save crops
        crop_path = f"{DEBUG_CROPS_DIR}/{img_path.stem}_plate{i+1}.jpg"
        cv2.imwrite(crop_path, crop)

        # Aggressive preprocessing
        processed = preprocess_plate(crop, aggressive=True)
        processed_path = f"{DEBUG_PROCESSED_DIR}/{img_path.stem}_plate{i+1}_proc.jpg"
        cv2.imwrite(processed_path, processed)

        # OCR pass 1 (strict)
        raw_text_strict = pytesseract.image_to_string(processed, config=STRICT_OCR).strip().replace(" ", "").upper()
        # OCR pass 2 (loose)
        raw_text_loose = pytesseract.image_to_string(processed, config=LOOSE_OCR).strip().replace(" ", "").upper()

        # If no valid match, try simple preprocessing
        if not re.search(PLATE_REGEX, raw_text_strict) and not re.search(PLATE_REGEX, raw_text_loose):
            processed = preprocess_plate(crop, aggressive=False)
            processed_path = f"{DEBUG_PROCESSED_DIR}/{img_path.stem}_plate{i+1}_proc_simple.jpg"
            cv2.imwrite(processed_path, processed)
            raw_text_strict = pytesseract.image_to_string(processed, config=STRICT_OCR).strip().replace(" ", "").upper()
            raw_text_loose = pytesseract.image_to_string(processed, config=LOOSE_OCR).strip().replace(" ", "").upper()

        match_strict = re.search(PLATE_REGEX, raw_text_strict)
        match_loose = re.search(PLATE_REGEX, raw_text_loose)

        if match_strict and len(match_strict.group(0)) in (9, 10):
            best_guess = match_strict.group(0)
            raw_best = raw_text_strict
            break
        elif match_loose and len(match_loose.group(0)) in (9, 10):
            best_guess = match_loose.group(0)
            raw_best = raw_text_loose
            break
        else:
            raw_best = raw_text_strict or raw_text_loose or ""

    # Create result line
    result_line = f"{img_path.name} → RAW:{raw_best} → CLEAN:{best_guess}"
    print(result_line)  # Show in terminal
    results_text.append(result_line)

# Save results to file
with open(RESULTS_FILE, "w") as f:
    f.write("\n".join(results_text))

print(f"\n✅ Results saved to {RESULTS_FILE}")
print(f"🔍 Cropped plates saved in '{DEBUG_CROPS_DIR}', processed plates in '{DEBUG_PROCESSED_DIR}'")


