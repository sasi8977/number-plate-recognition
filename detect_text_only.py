import os
import cv2
import numpy as np
import pytesseract
import easyocr
from ultralytics import YOLO
from datetime import datetime

# ==========================
# CONFIGURATION
# ==========================
MODEL_DIR = "runs/detect/train10/weights"
SAVE_DIR = "debug_crops"
PROC_DIR = "debug_processed"
RESULTS_FILE = "plate_text_results.txt"

# Create directories
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

# Initialize OCR engines
easyocr_reader = easyocr.Reader(['en'], gpu=False)

# Tesseract config: whitelist uppercase A-Z and digits 0-9
tess_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# ==========================
# FUNCTIONS
# ==========================

def preprocess_plate(image):
    """Apply preprocessing to improve OCR accuracy."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Noise removal but keep edges
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(thresh)
    return enhanced

def ocr_easyocr(image):
    """Run EasyOCR and return the result string."""
    results = easyocr_reader.readtext(image, detail=0, paragraph=False)
    return results[0] if results else ""

def ocr_tesseract(image):
    """Run Tesseract OCR and return the result string."""
    text = pytesseract.image_to_string(image, config=tess_config)
    return text.strip()

def clean_text(text):
    """Keep only A-Z and 0-9."""
    return ''.join([c for c in text.upper() if c.isalnum()])

def select_best(text1, text2):
    """Pick the better OCR result."""
    # Prefer the longer one with only valid characters
    t1 = clean_text(text1)
    t2 = clean_text(text2)
    if len(t1) >= len(t2):
        return t1 if t1 else t2
    else:
        return t2 if t2 else t1

# ==========================
# MAIN
# ==========================

def main():
    # Choose model
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]
    if not models:
        print("❌ No model found in", MODEL_DIR)
        return
    print("\nAvailable models:")
    for i, m in enumerate(models, 1):
        print(f"{i}. {m}")
    choice = input("\nSelect model number (press Enter for latest): ").strip()
    model_file = models[-1] if choice == "" else models[int(choice) - 1]
    model_path = os.path.join(MODEL_DIR, model_file)
    print(f"\n✅ Using model: {model_file}")

    # Load YOLO model
    model = YOLO(model_path)

    # Load test images
    images = [f for f in os.listdir() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        print("❌ No test images found in current directory.")
        return

    results_log = []
    for img_file in images:
        img = cv2.imread(img_file)
        if img is None:
            continue

        # Run detection
        detections = model(img)[0]
        plate_texts = []

        for i, box in enumerate(detections.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Save raw crop
            crop_path = os.path.join(SAVE_DIR, f"{os.path.splitext(img_file)[0]}_{i}.png")
            cv2.imwrite(crop_path, crop)

            # Preprocess crop
            processed = preprocess_plate(crop)
            proc_path = os.path.join(PROC_DIR, f"{os.path.splitext(img_file)[0]}_{i}.png")
            cv2.imwrite(proc_path, processed)

            # OCR both methods
            easy_text = ocr_easyocr(processed)
            tess_text = ocr_tesseract(processed)

            # Select best
            best_text = select_best(easy_text, tess_text)
            plate_texts.append(best_text if best_text else "UNREADABLE")

        # Save result for this image
        results_log.append(f"{img_file} → {', '.join(plate_texts)}")
        print(f"{img_file} → {', '.join(plate_texts)}")

    # Write to file
    with open(RESULTS_FILE, "w") as f:
        f.write("\n".join(results_log))
    print(f"\n✅ Results saved to {RESULTS_FILE}")
    print(f"🔍 Cropped plates saved in '{SAVE_DIR}', processed plates in '{PROC_DIR}'")

if __name__ == "__main__":
    main()

