import os
import re
import cv2
import pytesseract
import easyocr
import difflib
import numpy as np
from ultralytics import YOLO

# =====================
# USER SETTINGS
# =====================
MODEL_PATH = "/home/kishore/Downloads/number-plate-recognition-main/runs/detect/train10/weights/best.pt"
SOURCE_DIR = "/home/kishore/Downloads/test_images"
RESULTS_FILE = "plate_text_results.txt"
DEBUG_CROPS_DIR = "debug_crops"
DEBUG_PROC_DIR = "debug_proc"
PADDING = 20
PLATE_REGEX = r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$|^[A-Z]{2}\d{1,4}[A-Z]{1,2}\d{1,4}$' # Indian plate format

# Create debug folders
os.makedirs(DEBUG_CROPS_DIR, exist_ok=True)
os.makedirs(DEBUG_PROC_DIR, exist_ok=True)

# Load YOLO model & OCR reader
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'])


# =====================
# UTILITIES
# =====================
def cleanup_plate_text(text):
    """Fix common OCR mistakes like O/0, I/1."""
    corrections = {
        '0O': '0', 'O0': '0', 'O': '0',
        'I': '1', 'L': '1', 'Z': '2',
        'S': '5', 'B': '8'
    }
    cleaned = ""
    for ch in text:
        cleaned += corrections.get(ch, ch)
    return cleaned

def preprocess_plate(plate_img, aggressive=False):
    """Return preprocessed image for OCR."""
    plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Handle low-light images
    mean_brightness = np.mean(plate_gray)

    if mean_brightness < 60:  # very dark plate
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        plate_gray = clahe.apply(plate_gray)
    elif mean_brightness > 190:  # very bright plate
        plate_gray = cv2.convertScaleAbs(plate_gray, alpha=0.8, beta=-40)


    # Resize for better OCR accuracy
    plate_gray = cv2.resize(plate_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Contrast + brightness boost
    plate_gray = cv2.convertScaleAbs(plate_gray, alpha=1.6, beta=20)

    # Sharpen
    kernel_sharp = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])
    plate_gray = cv2.filter2D(plate_gray, -1, kernel_sharp)

    # Light blur to reduce pixel noise
    plate_gray = cv2.GaussianBlur(plate_gray, (1, 1), 0)

    if aggressive:
        # CLAHE (adaptive contrast enhancement)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        plate_gray = clahe.apply(plate_gray)

        # Median blur to remove speckles
        plate_gray = cv2.medianBlur(plate_gray, 3)

        # Adaptive threshold for high-contrast binarization
        plate_thresh = cv2.adaptiveThreshold(
            plate_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 41, 15
        )

        # Morphological closing + dilation
        kernel = np.ones((3, 3), np.uint8)
        plate_thresh = cv2.morphologyEx(plate_thresh, cv2.MORPH_CLOSE, kernel)
        plate_thresh = cv2.dilate(plate_thresh, kernel, iterations=1)

    else:
        # Light sharpen
        kernel_light = np.array([[0, -1, 0],
                                 [-1, 5, -1],
                                 [0, -1, 0]])
        plate_gray = cv2.filter2D(plate_gray, -1, kernel_light)

        # Gaussian blur
        plate_gray = cv2.GaussianBlur(plate_gray, (3, 3), 0)

        # Otsu's threshold
        _, plate_thresh = cv2.threshold(
            plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    return plate_thresh


def tesseract_ocr(image, psm=8, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"):
    """Run Tesseract OCR."""
    config = f"-c tessedit_char_whitelist={whitelist} --oem 3 --psm {psm}"
    text = pytesseract.image_to_string(image, config=config).strip()
    text = re.sub(r"[^A-Z0-9]", "", text.upper())
    return cleanup_plate_text(text)


def easyocr_ocr(image):
    """Run EasyOCR."""
    results = reader.readtext(image)
    if not results:
        return "", 0
    text, conf = results[0][1], results[0][2]
    text = re.sub(r"[^A-Z0-9]", "", text.upper())
    return cleanup_plate_text(text), conf


def regex_match_score(text):
    """Score text based on how closely it matches PLATE_REGEX."""
    if re.match(PLATE_REGEX, text):
        return 3  # Perfect match gets higher score
    elif len(text) >= 6:
        ratio = difflib.SequenceMatcher(None, text, "AB12CD1234").ratio()
        return 2 if ratio > 0.65 else 1
    return 0


def run_ocr_all(proc_img):
    """Run OCR in fallback order: EasyOCR -> Tesseract loose -> Tesseract strict."""
    candidates = []

    # EasyOCR first
    txt_easy, conf_easy = easyocr_ocr(proc_img)
    candidates.append(("EasyOCR", txt_easy, regex_match_score(txt_easy), conf_easy))

    # Tesseract loose
    txt_loose = tesseract_ocr(proc_img, psm=6)
    candidates.append(("TessLoose", txt_loose, regex_match_score(txt_loose), None))

    # Tesseract strict
    txt_strict = tesseract_ocr(proc_img, psm=8)
    candidates.append(("TessStrict", txt_strict, regex_match_score(txt_strict), None))

    # Tesseract PSM7 (single line mode)
    txt_psm7 = tesseract_ocr(proc_img, psm=7)
    candidates.append(("TessPSM7", txt_psm7, regex_match_score(txt_psm7), None))

    return candidates



def pick_best(candidates):
    """Pick best OCR result from candidates."""
    return sorted(
        candidates,
        key=lambda x: (x[2], x[3] if x[3] is not None else 0, -abs(len(x[1]) - 9)),
        reverse=True
    )[0]


def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading image: {img_path}")
        return None

    results = model.predict(
        source=img,
        conf=0.25,  # lower conf to catch faint plates
        iou=0.5,    # allow more overlaps
        imgsz=1280, # larger size for better small-object detection
        augment=True
    )[0]

    detections = results.boxes
    img_name = os.path.basename(img_path)
    best_plate_info = None

    for det_idx, det in enumerate(detections):
        conf = float(det.conf)
        if conf < 0.25:
            continue

        x1, y1, x2, y2 = map(int, det.xyxy[0])
        x1, y1 = max(0, x1 - PADDING - 5), max(0, y1 - PADDING - 5)
        x2, y2 = min(img.shape[1], x2 + PADDING + 5), min(img.shape[0], y2 + PADDING + 5)

        plate_crop = img[y1:y2, x1:x2]
        crop_name = f"{img_name}_plate{det_idx+1}.png"
        cv2.imwrite(os.path.join(DEBUG_CROPS_DIR, crop_name), plate_crop)
        
        # --- Always run both light & aggressive preprocessing ---
        proc_img_light = preprocess_plate(plate_crop, aggressive=False)
        proc_name_light = f"{img_name}_plate{det_idx+1}_light.png"
        cv2.imwrite(os.path.join(DEBUG_PROC_DIR, proc_name_light), proc_img_light)
        light_candidates = run_ocr_all(proc_img_light)

        proc_img_agg = preprocess_plate(plate_crop, aggressive=True)
        proc_name_agg = f"{img_name}_plate{det_idx+1}_agg.png"
        cv2.imwrite(os.path.join(DEBUG_PROC_DIR, proc_name_agg), proc_img_agg)
        aggressive_candidates = run_ocr_all(proc_img_agg)

        # Merge candidates & pick best
        all_candidates = light_candidates + aggressive_candidates
        best_candidate = pick_best(all_candidates)



        # Pick best candidate overall
        best_candidate = pick_best(all_candidates)

        if not best_plate_info or best_candidate[2] > best_plate_info[2]:
            best_plate_info = best_candidate + (conf,)

    if best_plate_info:
        return {
            "image": img_name,
            "method": best_plate_info[0],
            "plate_text": best_plate_info[1],
            "regex_score": best_plate_info[2],
            "ocr_conf": best_plate_info[3],
            "yolo_conf": best_plate_info[4]
        }
    return None


# =====================
# MAIN EXECUTION
# =====================
results_list = []
for file in os.listdir(SOURCE_DIR):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        res = process_image(os.path.join(SOURCE_DIR, file))
        if res:
            results_list.append(res)
            print(f"{res['image']}: {res['plate_text']} via {res['method']} (YOLO {res['yolo_conf']:.2f})")

with open(RESULTS_FILE, "w") as f:
    for r in results_list:
        f.write(f"{r['image']} | {r['plate_text']} | {r['method']} | YOLO: {r['yolo_conf']:.2f}\n")

print(f"Saved {len(results_list)} results to {RESULTS_FILE}")

