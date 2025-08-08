import cv2
import pytesseract
import re
from pathlib import Path
from ultralytics import YOLO
import os

# ---------------- CONFIG ---------------- #
MODEL_PATH = "last_copy.pt"  # Safe snapshot model
SOURCE_DIR = "/home/kishore/Downloads"   # Images folder
RESULTS_FILE = "plate_text_results_lastcopy.txt"
DEBUG_CROPS_DIR = "debug_crops"
DEBUG_PROCESSED_DIR = "debug_processed"
# ----------------------------------------- #

# Make debug dirs
os.makedirs(DEBUG_CROPS_DIR, exist_ok=True)
os.makedirs(DEBUG_PROCESSED_DIR, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)

# Indian number plate regex
PLATE_REGEX = r'[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}'

# Tesseract configs
STRICT_OCR = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
LOOSE_OCR = '--psm 7'

# Preprocessing for better OCR
def preprocess_plate(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Upscale for clarity
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Remove noise
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Binary threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2
    )

    # Morphological opening (clean small dots)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return thresh

results_text = []
image_paths = list(Path(SOURCE_DIR).glob("*.*"))

for img_path in image_paths:
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    detections = model(img)[0]
    best_guess = None

    for i, box in enumerate(detections.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Save raw crop
        crop_path = f"{DEBUG_CROPS_DIR}/{img_path.stem}_plate{i+1}.jpg"
        cv2.imwrite(crop_path, crop)

        processed = preprocess_plate(crop)

        # Save processed image
        processed_path = f"{DEBUG_PROCESSED_DIR}/{img_path.stem}_plate{i+1}_proc.jpg"
        cv2.imwrite(processed_path, processed)

        # First try strict OCR
        raw_text = pytesseract.image_to_string(processed, config=STRICT_OCR).strip().replace(" ", "").upper()

        # Fallback to loose OCR if strict fails
        if not re.search(PLATE_REGEX, raw_text):
            raw_text = pytesseract.image_to_string(processed, config=LOOSE_OCR).strip().replace(" ", "").upper()

        match = re.search(PLATE_REGEX, raw_text)
        if match:
            best_guess = match.group(0)
            break  # stop after finding valid plate
        elif best_guess is None:
            # Keep the first raw text as fallback if no regex match later
            best_guess = raw_text if raw_text else None

    final_text = best_guess if best_guess else "UNREADABLE"
    results_text.append(f"{img_path.name} → RAW:{raw_text} → CLEAN:{final_text}")
    print(f"{img_path.name} → RAW:{raw_text} → CLEAN:{final_text}")

with open(RESULTS_FILE, "w") as f:
    f.write("\n".join(results_text))

print(f"\n✅ Results saved to {RESULTS_FILE}")
print(f"🔍 Cropped plates saved in '{DEBUG_CROPS_DIR}', processed plates in '{DEBUG_PROCESSED_DIR}'")
