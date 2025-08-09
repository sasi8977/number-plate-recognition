import cv2
import pytesseract
import re
from pathlib import Path
from ultralytics import YOLO
import os

# ---------------- CONFIG ---------------- #
MODEL_PATH = "last_copy.pt"  # Your snapshot model
SOURCE_DIR = "/home/kishore/Downloads/kishore"   # Images folder
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
STRICT_OCR = '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --dpi 300'
LOOSE_OCR = '--psm 7 --oem 3 --dpi 300'

# Preprocessing
def preprocess_plate(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return thresh

# Get all images
image_paths = [p for p in Path(SOURCE_DIR).glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]

results_text = []

for img_path in image_paths:
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    detections = model(img)[0]
    best_guess = "UNREADABLE"
    raw_best = ""

    for i, box in enumerate(sorted(detections.boxes, key=lambda b: b.conf[0], reverse=True)):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Save crops
        crop_path = f"{DEBUG_CROPS_DIR}/{img_path.stem}_plate{i+1}.jpg"
        cv2.imwrite(crop_path, crop)

        processed = preprocess_plate(crop)
        processed_path = f"{DEBUG_PROCESSED_DIR}/{img_path.stem}_plate{i+1}_proc.jpg"
        cv2.imwrite(processed_path, processed)

        # OCR pass 1 (strict)
        raw_text_strict = pytesseract.image_to_string(processed, config=STRICT_OCR).strip().replace(" ", "").upper()
        # OCR pass 2 (loose)
        raw_text_loose = pytesseract.image_to_string(processed, config=LOOSE_OCR).strip().replace(" ", "").upper()

        match_strict = re.search(PLATE_REGEX, raw_text_strict)
        match_loose = re.search(PLATE_REGEX, raw_text_loose)

        if match_strict:
            best_guess = match_strict.group(0)
            raw_best = raw_text_strict
            break
        elif match_loose:
            best_guess = match_loose.group(0)
            raw_best = raw_text_loose
            break
        else:
            # Keep a non-matching raw text as fallback
            raw_best = raw_best or raw_text_strict or raw_text_loose

    results_text.append(f"{img_path.name} → RAW:{raw_best} → CLEAN:{best_guess}")
    print(f"{img_path.name} → RAW:{raw_best} → CLEAN:{best_guess}")

# Save results
with open(RESULTS_FILE, "w") as f:
    f.write("\n".join(results_text))

print(f"\n✅ Results saved to {RESULTS_FILE}")
print(f"🔍 Cropped plates saved in '{DEBUG_CROPS_DIR}', processed plates in '{DEBUG_PROCESSED_DIR}'")


