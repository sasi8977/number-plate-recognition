import cv2
import pytesseract
import re
from pathlib import Path
from ultralytics import YOLO
import os
from datetime import datetime

# ---------------- CONFIG ---------------- #
MODEL_PATH = "runs/detect/train10/weights/best.pt"  # Use your final best model
SOURCE_DIR = "/home/kishore/Downloads"              # Folder with test images
RESULTS_FILE = "plate_text_results.txt"
DEBUG_CROPS_DIR = "debug_crops"
DEBUG_PROCESSED_DIR = "debug_processed"
# ----------------------------------------- #

# Always start fresh
for folder in [DEBUG_CROPS_DIR, DEBUG_PROCESSED_DIR]:
    if os.path.exists(folder):
        for f in Path(folder).glob("*"):
            f.unlink()
    else:
        os.makedirs(folder, exist_ok=True)

if os.path.exists(RESULTS_FILE):
    os.remove(RESULTS_FILE)

# Load YOLO model
model = YOLO(MODEL_PATH)

# Regex for Indian number plates
PLATE_REGEX = r'[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{3,4}'

# OCR configs
STRICT_OCR = '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --dpi 300'
LOOSE_OCR = '--psm 6 --oem 3 --dpi 300'

# Preprocessing function
def preprocess_plate(crop, aggressive=True):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    if aggressive:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    else:
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    return thresh

# Get all images sorted by last modified time
image_paths = sorted(
    [p for p in Path(SOURCE_DIR).glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")],
    key=lambda p: p.stat().st_mtime,
    reverse=False
)

if not image_paths:
    print(f"❌ No images found in {SOURCE_DIR}")
    exit()

results_text = []

for img_path in image_paths:
    img_time = datetime.fromtimestamp(img_path.stat().st_mtime)
    print(f"\n📂 Processing: {img_path} (Last Modified: {img_time})")

    img = cv2.imread(str(img_path))
    if img is None:
        print("❌ Could not read image.")
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

        crop_path = f"{DEBUG_CROPS_DIR}/{img_path.stem}_plate{i+1}.jpg"
        cv2.imwrite(crop_path, crop)

        # Pass 1: aggressive preprocessing
        processed = preprocess_plate(crop, aggressive=True)
        processed_path = f"{DEBUG_PROCESSED_DIR}/{img_path.stem}_plate{i+1}_proc.jpg"
        cv2.imwrite(processed_path, processed)

        # OCR attempts
        ocr_attempts = [
            pytesseract.image_to_string(processed, config=STRICT_OCR).strip().replace(" ", "").upper(),
            pytesseract.image_to_string(processed, config=LOOSE_OCR).strip().replace(" ", "").upper()
        ]

        # Pass 2: simple preprocessing
        if not any(re.search(PLATE_REGEX, text) for text in ocr_attempts):
            processed_simple = preprocess_plate(crop, aggressive=False)
            processed_path = f"{DEBUG_PROCESSED_DIR}/{img_path.stem}_plate{i+1}_proc_simple.jpg"
            cv2.imwrite(processed_path, processed_simple)
            ocr_attempts.extend([
                pytesseract.image_to_string(processed_simple, config=STRICT_OCR).strip().replace(" ", "").upper(),
                pytesseract.image_to_string(processed_simple, config=LOOSE_OCR).strip().replace(" ", "").upper()
            ])

        # Fallback: inverted image OCR
        if not any(re.search(PLATE_REGEX, text) for text in ocr_attempts):
            inverted = cv2.bitwise_not(processed)
            ocr_attempts.append(
                pytesseract.image_to_string(inverted, config=LOOSE_OCR).strip().replace(" ", "").upper()
            )

        # Select best match
        for text in ocr_attempts:
            match = re.search(PLATE_REGEX, text)
            if match and len(match.group(0)) in (9, 10):
                best_guess = match.group(0)
                raw_best = text
                break

        if best_guess != "UNREADABLE":
            break
        else:
            raw_best = ocr_attempts[0] if ocr_attempts else ""

    # Print and store result
    print(f"{img_path.name} → RAW:{raw_best} → CLEAN:{best_guess}")
    results_text.append(f"{img_path.name} → RAW:{raw_best} → CLEAN:{best_guess}")

# Save all results
with open(RESULTS_FILE, "w") as f:
    f.write("\n".join(results_text))

print(f"\n✅ Results saved to {RESULTS_FILE}")
print(f"🔍 Cropped plates saved in '{DEBUG_CROPS_DIR}', processed plates in '{DEBUG_PROCESSED_DIR}'")



