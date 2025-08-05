# 🚗 Number Plate Recognition using YOLOv8 and OCR

A real-time number plate detection and recognition system built with **YOLOv8**, **EasyOCR**, and **Streamlit**.  
Detects and reads vehicle number plates from **images, videos, or live webcam feed** using a custom-trained model.

Developed by **Tummapudi Sasidhar**.

---

## ✅ Features

- Real-time number plate detection using YOLOv8
- OCR-based text recognition with EasyOCR
- Automatic fallback to pytesseract if EasyOCR fails
- Supports image, video, and webcam input
- Web-based interface built using Streamlit
- Error handling and OCR preprocessing built-in
- Trained on a custom dataset for Indian number plates

---

## 📁 Project Structure

```plaintext
number-plate-recognition/
│
├── detect_and_recognize.py     # Core detection + OCR script
├── streamlit_app.py            # Streamlit web app
├── preprocessing.py            # Script to split/train dataset
├── number-plate.yaml           # Dataset config for YOLOv8
├── training.ipynb              # Model training notebook

🚀 Installation
pip install -r requirements.txt

If you encounter issues with ultralytics, install it separately:

pip install ultralytics

🧪 How to Use
🔹 Run from Command Line
Detect and recognize number plates from an image:

bash
python detect_and_recognize.py --source path/to/image.jpg

From a webcam:

bash
python detect_and_recognize.py --source 0

From a video file:

bash
python detect_and_recognize.py --source path/to/video.mp4

🔹 Launch the Streamlit Web App
Run the app:
bash
streamlit run streamlit_app.py

Upload an image or video
Or enable webcam capture
View the processed image with bounding boxes and recognized text

🔧 Installation Guide (For Setup on Any System)
To run this Number Plate Recognition project locally, follow these steps:

✅ 1. Install Python
Make sure Python 3.8 or higher is installed.

2. Clone This Repository

git clone https://github.com/sasi8977/number-plate-recognition.git
cd number-plate-recognition

3. Install Python Dependencies
Use the provided requirements.txt file:

pip install -r requirements.txt
This installs all necessary libraries including:

ultralytics (YOLOv8)
easyocr
pytesseract
opencv-python
torch
streamlit (if using the web app
and more

4. Install Tesseract-OCR (Required for Text Recognition)
This is needed for pytesseract fallback OCR.

Windows:
Download Tesseract OCR and add its installation folder to your system PATH
(e.g., C:\Program Files\Tesseract-OCR)

Ubuntu:
sudo apt update
sudo apt install tesseract-ocr

macOS:
brew install tesserac

5. Place the Trained YOLOv8 Model
Make sure your trained model (best.pt) is located at:

swift
runs/detect/train/weights/best.pt
This is required for the detection script and the Streamlit app.

6. (Optional) Webcam Support
If you want to use live webcam input, ensure:
A camera is connected to your system
OpenCV has permission to access it
The script will automatically activate webcam mode when file_path = "webcam" is used.

👤 Author
Tummapudi Sasidhar

📄 License
This project is licensed under the MIT License.
You are free to use, modify, and distribute this code for personal or commercial use.

⭐️ Star this repo
If you find this project useful, consider giving it a ⭐️ on GitHub to support the work.


