import easyocr
from ultralytics import YOLO
import cv2
import os
import re

# ===== SETTINGS =====
MODEL_PATH = "runs/detect/train10/weights/best.pt"  # Path to your trained YOLO model
SOURCE_FOLDER = "/home/kishore/Downloads"           # Folder with test images
OUTPUT_TEXT_FILE = "plate_text_results.txt"         # File to save plate numbers
CROPS_FOLDER = "plate_crops"                        # Folder to store cropped plates
# ====================

# Create output folders
os.makedirs(CROPS_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)

# Load EasyOCR reader
reader = easyocr.Reader(['en'])

# Function to clean OCR result for Indian plates
def clean_plate_text(text):
    # Remove spaces and unwanted chars
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    # Match Indian format (e.g., TN87C5106)
    match = re.match(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{1,4}$', text)
    return match.group(0) if match else text

# Open text file for writing results
with open(OUTPUT_TEXT_FILE, "w") as f:
    # Run YOLO prediction
    results = model.predict(source=SOURCE_FOLDER, save=False, conf=0.5)

    for img_i, result in enumerate(results):
        img_path = result.path
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)

        # Loop through detected boxes
        for j, box in enumerate(result.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            crop = img[y1:y2, x1:x2]

            # Save cropped plate
            crop_path = os.path.join(CROPS_FOLDER, f"{img_name}_plate{j+1}.jpg")
            cv2.imwrite(crop_path, crop)

            # OCR on the cropped plate
            ocr_result = reader.readtext(crop)
            if ocr_result:
                detected_text = " ".join([res[1] for res in ocr_result])
                cleaned_text = clean_plate_text(detected_text)
            else:
                cleaned_text = "NO_TEXT_FOUND"

            # Write to file
            f.write(f"{img_name} → {cleaned_text}\n")
            print(f"{img_name} → {cleaned_text}")

print(f"\n✅ Results saved to {OUTPUT_TEXT_FILE}")
print(f"✅ Cropped plates saved to {CROPS_FOLDER}/")
