import os
import cv2
import pytesseract
import re
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

# ---------------- CONFIG ---------------- #
MODEL_PATH = "last_copy.pt"
SOURCE_DIR = "/home/kishore/Downloads"
RESULTS_FILE = "plate_text_results.txt"
DEBUG_CROPS_DIR = "debug_crops"
DEBUG_PROCESSED_DIR = "debug_processed"
# ----------------------------------------- #

# Clear old results file
open(RESULTS_FILE, "w").close()

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

# Get all images
image_paths = [p for p in Path(SOURCE_DIR).glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]

if not image_paths:
    print(f"❌ No images found in {SOURCE_DIR}")
    exit()

results_text = []

for img_path in image_paths:
    # Debug print for path + last modified
    mod_time = datetime.fromtimestamp(os.path.getmtime(img_path)).strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n📂 Processing: {img_path.resolve()} (Last Modified: {mod_time})")

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    detections = model(img)[0]
    best_guess = "UNREADABLE"
    raw_best = ""
    ...
    # (rest of your detection & OCR code stays same)

