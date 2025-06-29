import os
from twilio.rest import Client
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

# Load environment variables from a .env file
load_dotenv()

TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_FROM")
recipients = os.getenv("TWILIO_TO").split(',')

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QTextEdit, QCheckBox, QHBoxLayout
)
import sys
import cv2
import numpy as np
import datetime
import pygame
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import QTimer, Qt
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import mediapipe as mp

base_dir = os.path.dirname(os.path.abspath(__file__))

class SafetyDetector:
    def __init__(self, alert_callback=None):
        yolo_path = os.path.join(base_dir, "yolov8n.pt")
        self.detector = YOLO(yolo_path)
        model_path = os.path.join(base_dir, "models", "gender_model.h5")
        self.gender_model = load_model(model_path)
        self.gender_labels = {0: "Male", 1: "Female"}
        self.CONF_THRESHOLD = 0.8
        self.INPUT_SIZE = (64, 64)
        self.alert_callback = alert_callback

        pygame.mixer.init()
        alert_path = os.path.join(base_dir, "alert.wav")
        self.alert_sound = pygame.mixer.Sound(alert_path)
        self.alert_channel = None

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.lone_woman_alert_triggered = False

        self.enable_gender_detection = True
        self.enable_lone_woman_detection = True
        self.enable_surrounding_detection = True
        self.enable_sos_detection = True

        self.alert_muted = False
        self.alert_timer = QTimer()
        self.alert_timer.setInterval(10000)
        self.alert_timer.setSingleShot(True)
        self.alert_timer.timeout.connect(self.reenable_alerts)

    def preprocess_face(self, face_img):
        face_img = cv2.resize(face_img, self.INPUT_SIZE)
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        return np.expand_dims(face_img, axis=0) / 255.0
    
    def send_sms_alert(self, reason):
        if TWILIO_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM and recipients:
            try:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
                for number in recipients:
                    message = client.messages.create(
                        from_=TWILIO_FROM,
                        to=number,
                        body=f"SHASHTRA ALERT:\n{reason} \nTime: {current_time}"
                    )
                    print(f"[SMS SENT] To {number} | SID: {message.sid}")
            except Exception as e:
                print(f"[SMS ERROR] {e}")
        else:
            print("[SMS ERROR] Twilio credentials or recipient list missing.")


    def play_alert(self, reason):
        if self.alert_muted:
            return
        print(f"[ALERT] {reason}")
        if self.alert_callback:
            self.alert_callback(True, reason) 
        if not self.alert_channel or not self.alert_channel.get_busy():
            self.alert_channel = self.alert_sound.play()

    def stop_alert(self):
        if self.alert_channel:
            self.alert_channel.stop()
        self.alert_muted = True
        self.alert_timer.start()

    def reenable_alerts(self):
        self.alert_muted = False

    def detect_sos_gesture(self, frame, female_count, male_count):
        if female_count < 1 or male_count > 0:
            return frame  # Do not trigger SOS unless only females are present

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_up = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y < landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y
            right_up = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y < landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            if (left_up and not right_up) or (right_up and not left_up):
                self.play_alert("SOS Gesture Detected")
                self.send_sms_alert("SOS Gesture Detected")
                cv2.putText(frame, "SOS Gesture Detected!", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        return frame


    def analyze_frame(self, frame):
        results = self.detector(frame, classes=0, conf=self.CONF_THRESHOLD)
        male_count, female_count = 0, 0

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_roi = frame[y1:y2, x1:x2]
                try:
                    processed_face = self.preprocess_face(face_roi)
                    pred = self.gender_model.predict(processed_face, verbose=0)
                    gender = self.gender_labels[int(np.argmax(pred))]
                    if gender == "Male":
                        male_count += 1
                        color = (255, 0, 0)
                    else:
                        female_count += 1
                        color = (255, 105, 180)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, gender, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                except Exception as e:
                    print("Face analysis error:", e)

        current_hour = datetime.datetime.now().hour

        if self.enable_lone_woman_detection:
            if 20 <= current_hour or current_hour <= 5:
                if female_count == 1 and male_count > 0:
                    if not self.lone_woman_alert_triggered:
                        self.play_alert("Lone woman at night")
                        self.lone_woman_alert_triggered = True
                        cv2.putText(frame, "\u26a0\ufe0f Lone Woman at Night!", (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    self.lone_woman_alert_triggered = False

        if self.enable_surrounding_detection:
            if female_count == 1 and male_count >= 5:
                self.play_alert("Woman surrounded by men")
                cv2.putText(frame, "\u26a0\ufe0f Surrounded Woman!", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 3)

        if self.enable_sos_detection:
            frame = self.detect_sos_gesture(frame, female_count, male_count)

        return frame, male_count, female_count

class GenderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SHASHTRA - A Women Protection System")
        self.setStyleSheet("background-color: #121212; color: white; font-family: 'Segoe UI';")

        self.detector = SafetyDetector(alert_callback=self.update_alert_status)

        self.title = QLabel("SHASHTRA - A Women Protection System")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        self.title.setStyleSheet("color: #FF69B4; padding: 0px; background-color: #1e1e1e; border-radius: 12px;")

        self.image_label_1 = QLabel("Webcam 1 Feed")
        self.image_label_1.setFixedSize(700, 500)
        self.image_label_1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label_1.setStyleSheet("background-color: #2b2b2b; border: 3px solid #FF69B4; border-radius: 10px;")

        self.image_label_2 = QLabel("Webcam 2 Feed")
        self.image_label_2.setFixedSize(700, 500)
        self.image_label_2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label_2.setStyleSheet("background-color: #2b2b2b; border: 3px solid #FF69B4; border-radius: 10px;")

        self.result_text = QTextEdit()
        self.result_text.setFixedHeight(70)
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("""
            background-color: #1e1e1e;
            border: 2px solid #FF69B4;
            border-radius: 8px;
            padding: 10px;
            font-size: 14px;
        """)

        self.alert_label = QLabel(" Status: Safe")
        self.alert_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.alert_label.setStyleSheet("color: white; background-color: green; padding: 10px; border-radius: 8px;")

        self.btn_webcam_1 = QPushButton("Start Webcam 1")
        self.btn_webcam_2 = QPushButton("Start Webcam 2")
        self.btn_stop_alert = QPushButton("Stop Alert")

        self.checkbox_sos = QCheckBox("Enable SOS Detection")
        self.checkbox_lone = QCheckBox("Enable Lone Woman Detection")
        self.checkbox_sos.setChecked(True)
        self.checkbox_lone.setChecked(True)

        self.checkbox_sos.setStyleSheet("QCheckBox { font-size: 18px; padding: 10px; }")
        self.checkbox_lone.setStyleSheet("QCheckBox { font-size: 18px; padding: 10px; }")

        for btn in [self.btn_webcam_1, self.btn_webcam_2, self.btn_stop_alert]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #FF69B4;
                    color: #000;
                    font-weight: bold;
                    padding: 10px 20px;
                    border-radius: 10px;
                    font-size: 18px;
                }
                QPushButton:hover {
                    background-color: #ff85c1;
                }
            """)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.title)

        center_layout = QHBoxLayout()
        center_layout.addStretch()
        center_layout.addWidget(self.image_label_1)
        center_layout.addWidget(self.image_label_2)
        center_layout.addStretch()
        main_layout.addLayout(center_layout)

        main_layout.addSpacing(10)
        main_layout.addWidget(self.result_text)
        main_layout.addWidget(self.alert_label)

        btn_row_layout = QHBoxLayout()
        btn_row_layout.addStretch()
        btn_row_layout.addWidget(self.btn_webcam_1)
        btn_row_layout.addWidget(self.btn_webcam_2)
        btn_row_layout.addWidget(self.btn_stop_alert)
        btn_row_layout.addStretch()

        checkbox_layout = QHBoxLayout()
        checkbox_layout.addStretch()
        checkbox_layout.addWidget(self.checkbox_sos)
        checkbox_layout.addWidget(self.checkbox_lone)
        checkbox_layout.addStretch()

        main_layout.addLayout(btn_row_layout)
        main_layout.addLayout(checkbox_layout)

        self.setLayout(main_layout)

        self.btn_webcam_1.clicked.connect(self.start_webcam_1)
        self.btn_webcam_2.clicked.connect(self.start_webcam_2)
        self.btn_stop_alert.clicked.connect(self.stop_alert_clicked)
        self.checkbox_sos.stateChanged.connect(self.toggle_sos)
        self.checkbox_lone.stateChanged.connect(self.toggle_lone_woman)

        self.cap_1 = None
        self.cap_2 = None
        self.timer = QTimer()
        self.alert_triggered = False  # Track if alert is active
        self.timer.timeout.connect(self.update_frame)

    def toggle_sos(self):
        self.detector.enable_sos_detection = self.checkbox_sos.isChecked()

    def toggle_lone_woman(self):
        self.detector.enable_lone_woman_detection = self.checkbox_lone.isChecked()

    def stop_alert_clicked(self):
        self.detector.stop_alert()
        self.update_alert_status(False, "Safe")

    def start_webcam_1(self):
        if self.cap_1:
            self.cap_1.release()
        self.cap_1 = cv2.VideoCapture(0)
        self.timer.start(30)

    def start_webcam_2(self):
        if self.cap_2:
            self.cap_2.release()
        self.cap_2 = cv2.VideoCapture(1)
        self.timer.start(30)

    def update_frame(self):
        if (self.cap_1 and self.cap_1.isOpened()) or (self.cap_2 and self.cap_2.isOpened()):
            if self.cap_1 and self.cap_1.isOpened():
                ret_1, frame_1 = self.cap_1.read()
                if ret_1:
                    processed_frame_1, males_1, females_1 = self.detector.analyze_frame(frame_1)
                    self.display_image(processed_frame_1, self.image_label_1)
                else:
                    males_1 = females_1 = 0
            else:
                males_1 = females_1 = 0

            if self.cap_2 and self.cap_2.isOpened():
                ret_2, frame_2 = self.cap_2.read()
                if ret_2:
                    processed_frame_2, males_2, females_2 = self.detector.analyze_frame(frame_2)
                    self.display_image(processed_frame_2, self.image_label_2)
                else:
                    males_2 = females_2 = 0
            else:
                males_2 = females_2 = 0

            self.result_text.setText(f"Webcam 1 - Males: {males_1}, Females: {females_1} | "
                                     f"Webcam 2 - Males: {males_2}, Females: {females_2}")

            if not self.alert_triggered:
                if males_1 >= 5 or males_2 >= 5:
                    self.update_alert_status(True, "Woman surrounded by men")
                else:
                    self.update_alert_status(False, "Safe")

    def update_alert_status(self, alert_active, reason):
        self.alert_triggered = alert_active
        if alert_active:
            self.alert_label.setText(f" ALERT: {reason}")
            self.alert_label.setStyleSheet("color: white; background-color: red; padding: 10px; border-radius: 8px; font-size: 18px;")
        else:
            self.alert_label.setText(" Status: Safe")
            self.alert_label.setStyleSheet("color: white; background-color: green; padding: 10px; border-radius: 8px; font-size: 18px;")


    def display_image(self, frame, label):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(700, 500, Qt.AspectRatioMode.KeepAspectRatio)
        label.setPixmap(pixmap)

    def closeEvent(self, event):
        if self.cap_1 and self.cap_1.isOpened():
            self.cap_1.release()
        if self.cap_2 and self.cap_2.isOpened():
            self.cap_2.release()
        self.timer.stop()
        self.detector.stop_alert()
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = GenderApp()
    win.resize(900, 900)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 