import cv2
import torch
import pytesseract
import re
from pathlib import Path
from ultralytics import YOLO
import os

# ---------------- CONFIG ---------------- #
WEIGHTS_DIR = Path("runs/detect/train10/weights")  # folder containing .pt files
SOURCE_DIR = "/home/kishore/Downloads"             # folder containing images
# ----------------------------------------- #

# Find all available model files
model_files = sorted(
    WEIGHTS_DIR.glob("*.pt"),
    key=os.path.getmtime,
    reverse=True
)

if not model_files:
    raise FileNotFoundError(f"No .pt model files found in {WEIGHTS_DIR}")

print("\nAvailable models:")
for i, mf in enumerate(model_files, start=1):
    print(f"{i}. {mf.name}")

choice = input("\nSelect model number (press Enter for latest): ").strip()

if choice == "":
    chosen_model = model_files[0]  # latest model
else:
    try:
        chosen_model = model_files[int(choice) - 1]
    except (ValueError, IndexError):
        raise ValueError("Invalid choice.")

print(f"\n✅ Using model: {chosen_model.name}")

# Results file will be named according to model
RESULTS_FILE = f"plate_text_results_{chosen_model.stem}.txt"

# Load YOLO model
model = YOLO(str(chosen_model))

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

