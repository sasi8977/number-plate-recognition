import os
import cv2
import re
import torch
import easyocr
import pytesseract
import numpy as np
from pathlib import Path
from PIL import Image

# =====================
# USER SETTINGS
# =====================
MODEL_PATH = "/home/kishore/Downloads/number-plate-recognition-main/runs/detect/train10/weights/best.pt"
SOURCE_DIR = "/home/kishore/Downloads/test_images"
RESULTS_FILE = "plate_text_results.txt"
DEBUG_CROPS_DIR = "debug_crops"
DEBUG_PROC_DIR = "debug_proc"
PADDING = 15
PLATE_REGEX = r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{1,4}$'  # Indian plate format

# Create output folders
os.makedirs(DEBUG_CROPS_DIR, exist_ok=True)
os.makedirs(DEBUG_PROC_DIR, exist_ok=True)

# Load YOLO model
model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH)
model.conf = 0.25  # Lowered threshold for more detections

# Load OCR engines
easy_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

def preprocess_for_ocr(crop):
    """Preprocess image for sharper OCR results with contrast boost."""
    # Convert to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Slight blur to reduce noise, but keep edges
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Sharpen edges (extra clarity for characters)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)

    # Increase contrast
    gray = cv2.equalizeHist(gray)

    # Adaptive threshold for lighting variation
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 15
    )

    # Invert so text is black on white (OCR prefers this)
    thresh = cv2.bitwise_not(thresh)

    return thresh


def run_easyocr(img):
    results = easy_reader.readtext(img)
    if results:
        return max(results, key=lambda x: x[2])[1].strip().upper()
    return ""

def run_tesseract(img):
    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(img, config=config)
    return text.strip().upper()

def score_plate(text):
    return bool(re.match(PLATE_REGEX, text))

def process_image(image_path):
    img_name = os.path.basename(image_path)
    img = cv2.imread(image_path)

    # YOLO detection
    results = model(img)
    detections = results.xyxy[0].cpu().numpy()

    if len(detections) == 0:
        return f"{img_name} | NO_PLATE | YOLO: 0.00"

    best_conf = 0
    best_plate = "NO_PLATE"

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        conf = float(conf)
        if conf > best_conf:
            # Apply padding
            x1 = max(0, int(x1) - PADDING)
            y1 = max(0, int(y1) - PADDING)
            x2 = min(img.shape[1], int(x2) + PADDING)
            y2 = min(img.shape[0], int(y2) + PADDING)

            plate_crop = img[y1:y2, x1:x2]
            cv2.imwrite(f"{DEBUG_CROPS_DIR}/{img_name}", plate_crop)

            proc_img = preprocess_for_ocr(plate_crop)
            cv2.imwrite(f"{DEBUG_PROC_DIR}/{img_name}", proc_img)

            # Run both OCR engines
            easy_text = run_easyocr(proc_img)
            tess_text = run_tesseract(proc_img)

            candidates = [easy_text, tess_text]
            candidates = [c for c in candidates if c and score_plate(c)]

            if candidates:
                best_plate = max(candidates, key=len)
            else:
                best_plate = easy_text or tess_text or "NO_PLATE"

            best_conf = conf

    return f"{img_name} | {best_plate} | YOLO: {best_conf:.2f}"

if __name__ == "__main__":
    results_list = []
    for file in sorted(Path(SOURCE_DIR).glob("*")):
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            results_list.append(process_image(str(file)))

    with open(RESULTS_FILE, "w") as f:
        for line in results_list:
            print(line)
            f.write(line + "\n")
