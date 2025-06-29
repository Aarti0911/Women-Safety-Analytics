from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Load YOLOv8 person detection model
detection_model = YOLO("models/yolov8n.pt")

# Load gender classification model
gender_model = load_model("models/gender_model.h5")