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

👤 Author
Tummapudi Sasidhar

📄 License
This project is licensed under the MIT License.
You are free to use, modify, and distribute this code for personal or commercial use.

⭐️ Star this repo
If you find this project useful, consider giving it a ⭐️ on GitHub to support the work.


