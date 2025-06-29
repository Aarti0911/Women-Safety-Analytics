SHASHTRA - A Women Protection System 

This is a real-time women safety monitoring system that uses AI and computer vision technologies to detect persons, classify gender, and identify SOS gestures. It can trigger alerts and send SMS messages in emergency situations.

Features

- Real-time **person detection** using YOLOv8
- **Gender classification** using a CNN (`gender_model.h5`)
- **SOS gesture detection** using MediaPipe Pose (one hand raised)
- Alerts through:
  - On-screen warning
  - Audio (`alert.wav`)
  - SMS using Twilio (if integrated)
- Dual webcam support (`webcam_one.py` and `webcam_two.py`)

---

Project Structure

| File | Description |
|------|-------------|
| alert.wav | Alert sound played when SOS is detected |
| detection_model.py | Logic for detecting gender and people in the frame |
| download_model.py | Utility to download necessary models (if required) |
| gender_model.h5 | Pretrained Keras model for gender classification |
| load_model.py | Loads the gender classification model |
| person_detection.py | YOLOv8 person detection wrapper |
| train.py` | General training script (if any) |
| train_gendermodel.py | Training script for the gender classification model |
| webcam_one.py | Main script to run detection from Webcam 1 |
| webcam_two.py | Main script to run detection from Webcam 1 and 2 |
| yolov8n.pt | YOLOv8 Nano weights used for object/person detection |

---

Requirements

Make sure the following libraries are installed:

pip install opencv-python mediapipe tensorflow torch ultralytics numpy
If you are using Twilio for SMS alerts, install:

pip install twilio

How to Run
Clone the repo or download the files

Download YOLOv8 weights (if not present):
yolo download model=yolov8n
Train Gender Model (Optional if gender_model.h5 is already trained):

python train_gendermodel.py
Run the application:

For single webcam:
python webcam_one.py

For dual webcam setup:
python webcam_two.py

Gesture Detection Logic
The SOS gesture is detected only for women (female_count ≥ 1 and male_count = 0) and triggers when only one hand is raised above the shoulder.

Model Summary
YOLOv8: Used for person detection (yolov8n.pt)

Gender Classification: Custom CNN model trained and saved as gender_model.h5

MediaPipe: Used for detecting body pose landmarks (for SOS gestures)

Optional: Twilio SMS Setup
If you're using Twilio to send SMS alerts, add your credentials in the appropriate file (typically in detection_model.py or a separate config file).

Webcam Notes
Webcam feed is accessed using OpenCV.

If you're using an external webcam, adjust the index (0 or 1) in cv2.VideoCapture().

Alert System
When SOS is detected:

An audio alert (alert.wav) is played.

A red on-screen warning is displayed.

(Optional) SMS is sent via Twilio.

Final Note
This system is developed for enhancing the safety of women in public spaces using AI-based detection and alert mechanisms. It’s best run in real-time environments where immediate responses are crucial.
