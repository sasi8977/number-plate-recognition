import cv2
import pytesseract
import re
from pathlib import Path
from ultralytics import YOLO
import os

# ---------------- CONFIG ---------------- #
MODEL_PATH = "runs/detect/train10/weights/best.pt"  # Updated to your latest best model
SOURCE_DIR = "/home/kishore/Downloads/kishore"
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
PLATE_REGEX = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$'

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
        if box.conf[0] < 0.5:  # Skip low-confidence detections
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Save crops
        cv2.imwrite(f"{DEBUG_CROPS_DIR}/{img_path.stem}_plate{i+1}.jpg", crop)

        # Try aggressive preprocessing
        processed = preprocess_plate(crop, aggressive=True)
        cv2.imwrite(f"{DEBUG_PROCESSED_DIR}/{img_path.stem}_plate{i+1}_proc.jpg", processed)

        # OCR pass 1 (strict)
        raw_text_strict = pytesseract.image_to_string(processed, config=STRICT_OCR).strip().replace(" ", "").upper()
        # OCR pass 2 (loose)
        raw_text_loose = pytesseract.image_to_string(processed, config=LOOSE_OCR).strip().replace(" ", "").upper()

        # Try simple preprocessing if no match
        if not re.fullmatch(PLATE_REGEX, raw_text_strict) and not re.fullmatch(PLATE_REGEX, raw_text_loose):
            processed = preprocess_plate(crop, aggressive=False)
            cv2.imwrite(f"{DEBUG_PROCESSED_DIR}/{img_path.stem}_plate{i+1}_proc_simple.jpg", processed)
            raw_text_strict = pytesseract.image_to_string(processed, config=STRICT_OCR).strip().replace(" ", "").upper()
            raw_text_loose = pytesseract.image_to_string(processed, config=LOOSE_OCR).strip().replace(" ", "").upper()

        # Validate & select best
        if re.fullmatch(PLATE_REGEX, raw_text_strict):
            best_guess = raw_text_strict
            raw_best = raw_text_strict
            break
        elif re.fullmatch(PLATE_REGEX, raw_text_loose):
            best_guess = raw_text_loose
            raw_best = raw_text_loose
            break
        else:
            raw_best = raw_text_strict or raw_text_loose or ""

    # Save & print result
    result_line = f"{img_path.name} → RAW:{raw_best} → CLEAN:{best_guess}"
    results_text.append(result_line)
    print(result_line)  # Live output in terminal

# Save results to file
with open(RESULTS_FILE, "w") as f:
    f.write("\n".join(results_text))

print(f"\n✅ Results saved to {RESULTS_FILE}")
print(f"🔍 Cropped plates saved in '{DEBUG_CROPS_DIR}', processed plates in '{DEBUG_PROCESSED_DIR}'")



