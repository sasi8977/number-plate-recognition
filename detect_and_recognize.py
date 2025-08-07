from ultralytics import YOLO
from easyocr import Reader
import pytesseract
import torch
import cv2
import os
import csv
import time

# --- Check if best.pt model exists ---
if not os.path.exists('runs/detect/train/weights/best.pt'):
    print("❌ Error: best.pt model not found! Please train the model first.")
    exit(1)

CONFIDENCE_THRESHOLD = 0.4
COLOR = (0, 255, 0)

def detect_number_plates(image, model, display=False):
    start = time.time()
    detections = model.predict(image)[0].boxes.data

    if detections.shape != torch.Size([0, 6]):
        boxes = []
        confidences = []

        for detection in detections:
            confidence = detection[4]
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue
            boxes.append(detection[:4])
            confidences.append(confidence)

        print(f"{len(boxes)} Number plate(s) detected.")
        number_plate_list = []

        for i in range(len(boxes)):
            xmin, ymin, xmax, ymax = map(int, boxes[i])
            number_plate_list.append([[xmin, ymin, xmax, ymax]])

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), COLOR, 2)
            text = "Number Plate: {:.2f}%".format(confidences[i] * 100)
            cv2.putText(image, text, (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)

            if display:
                number_plate = image[ymin:ymax, xmin:xmax]
                cv2.imshow(f"Number plate {i}", number_plate)

        print(f"Detection time: {(time.time() - start) * 1000:.0f} ms")
        return number_plate_list
    else:
        print("No number plates detected.")
        return []

def recognize_number_plates(image_or_path, reader, number_plate_list, write_to_csv=False):
    start = time.time()

    image = cv2.imread(image_or_path) if isinstance(image_or_path, str) else image_or_path

    for i, box in enumerate(number_plate_list):
        np_image = image[box[0][1]:box[0][3], box[0][0]:box[0][2]]
        gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        detection = reader.readtext(thresh, paragraph=True)

        if len(detection) == 0:
            text = pytesseract.image_to_string(thresh, config='--psm 7').strip()
        else:
            text = str(detection[0][1])

        number_plate_list[i].append(text)

    if write_to_csv:
        with open("number_plates.csv", "w", newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["image_path", "box", "text"])
            for box, text in number_plate_list:
                csv_writer.writerow([image_or_path, box, text])

    print(f"Recognition time: {(time.time() - start) * 1000:.0f} ms")
    return number_plate_list

# Example usage if you run this file directly (for testing)
if __name__ == "__main__":
    model = YOLO("runs/detect/train/weights/best.pt")
    reader = Reader(['en'], gpu=True)

    file_path = "datasets/images/test/0fc216ca-131.jpg"
    image = cv2.imread(file_path)
    number_plate_list = detect_number_plates(image, model, display=True)
    
    if number_plate_list:
        number_plate_list = recognize_number_plates(image, reader, number_plate_list, write_to_csv=True)
        for box, text in number_plate_list:
            cv2.putText(image, text, (box[0], box[3] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)
        cv2.imshow('Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

