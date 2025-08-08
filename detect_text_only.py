import cv2
import torch
import pytesseract
import re
from pathlib import Path
from ultralytics import YOLO

# ---------------- CONFIG ---------------- #
MODEL_PATH = "runs/detect/train10/weights/best.pt"  # your trained YOLO model
SOURCE_DIR = "/home/kishore/Downloads"              # folder containing images
RESULTS_FILE = "plate_text_results.txt"
# ----------------------------------------- #

# Load YOLO model
model = YOLO(MODEL_PATH)

# Indian number plate regex
PLATE_REGEX = r'[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}'

# Pytesseract config
OCR_CONFIG = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# Preprocessing function
def preprocess_plate(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2
    )
    return thresh

# Detect plates and extract text
results_text = []
image_paths = list(Path(SOURCE_DIR).glob("*.*"))

for img_path in image_paths:
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    detections = model(img)[0]

    plate_text_found = "NO_TEXT_FOUND"
    for box in detections.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Preprocess for OCR
        processed = preprocess_plate(crop)

        # OCR
        text = pytesseract.image_to_string(processed, config=OCR_CONFIG)
        text = text.strip().replace(" ", "").upper()

        # Regex filter for Indian plates
        match = re.search(PLATE_REGEX, text)
        if match:
            plate_text_found = match.group(0)
            break  # stop after first valid plate

    results_text.append(f"{img_path.name} → {plate_text_found}")
    print(f"{img_path.name} → {plate_text_found}")

# Save results
with open(RESULTS_FILE, "w") as f:
    f.write("\n".join(results_text))

print(f"\n✅ Results saved to {RESULTS_FILE}")
