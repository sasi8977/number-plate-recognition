import cv2
import torch
import pytesseract
import re
from pathlib import Path
from ultralytics import YOLO

# ---------------- CONFIG ---------------- #
MODEL_PATH = "runs/detect/train10/weights/best.pt"
SOURCE_DIR = "/home/kishore/Downloads"
RESULTS_FILE = "plate_text_results.txt"
SAVE_DEBUG_CROPS = True   # <-- NEW: Save cropped plate images for debugging
DEBUG_DIR = "debug_crops" # <-- folder for debug crops
# ----------------------------------------- #

# Load YOLO model
model = YOLO(MODEL_PATH)

# Relaxed regex (2 letters + numbers, optional letters in between)
PLATE_REGEX = r'[A-Z]{2}[0-9]{1,2}[A-Z0-9]{0,2}[0-9]{3,4}'

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

# Debug crop saving
def save_crop(image, name, idx):
    Path(DEBUG_DIR).mkdir(exist_ok=True)
    path = Path(DEBUG_DIR) / f"{name}_plate{idx}.jpg"
    cv2.imwrite(str(path), image)

# Detect plates and extract text
results_text = []
image_paths = list(Path(SOURCE_DIR).glob("*.*"))

for img_path in image_paths:
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    detections = model(img)[0]

    plate_texts = []  # store all possible plates for this image
    for i, box in enumerate(detections.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        if SAVE_DEBUG_CROPS:
            save_crop(crop, img_path.stem, i)

        processed = preprocess_plate(crop)

        text = pytesseract.image_to_string(processed, config=OCR_CONFIG)
        text = text.strip().replace(" ", "").upper()

        if re.search(PLATE_REGEX, text):
            plate_texts.append(re.search(PLATE_REGEX, text).group(0))
        elif text:  # keep OCR text even if regex fails
            plate_texts.append(f"RAW:{text}")

    if plate_texts:
        result = "; ".join(plate_texts)
    else:
        result = "NO_TEXT_FOUND"

    results_text.append(f"{img_path.name} → {result}")
    print(f"{img_path.name} → {result}")

with open(RESULTS_FILE, "w") as f:
    f.write("\n".join(results_text))

print(f"\n✅ Results saved to {RESULTS_FILE}")
if SAVE_DEBUG_CROPS:
    print(f"🔍 Cropped plates saved in '{DEBUG_DIR}'")

