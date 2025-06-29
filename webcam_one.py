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


class SafetyDetector:
    def __init__(self):
        self.detector = YOLO("yolov8n.pt")
        self.gender_model = load_model(r"C:\\Users\\HP\\women-safety-env\\models\\gender_model.h5")
        self.gender_labels = {0: "Male", 1: "Female"}
        self.CONF_THRESHOLD = 0.7
        self.INPUT_SIZE = (64, 64)

        pygame.mixer.init()
        self.alert_sound = pygame.mixer.Sound("alert.wav")
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
        self.alert_timer.setInterval(20000)
        self.alert_timer.setSingleShot(True)
        self.alert_timer.timeout.connect(self.reenable_alerts)

    def preprocess_face(self, face_img):
        face_img = cv2.resize(face_img, self.INPUT_SIZE)
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        return np.expand_dims(face_img, axis=0) / 255.0

    def play_alert(self, reason):
        if self.alert_muted:
            return
        print(f"[ALERT] {reason}")
        if not self.alert_channel or not self.alert_channel.get_busy():
            self.alert_channel = self.alert_sound.play()

    def stop_alert(self):
        if self.alert_channel:
            self.alert_channel.stop()
        self.alert_muted = True
        self.alert_timer.start()

    def reenable_alerts(self):
        self.alert_muted = False

    def detect_sos_gesture(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_up = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y < landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y
            right_up = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y < landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            if (left_up and not right_up) or (right_up and not left_up):
                self.play_alert("SOS Gesture Detected")
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
            frame = self.detect_sos_gesture(frame)

        return frame, male_count, female_count


class GenderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SHASHTRA - A Women Protection System")
        self.setStyleSheet("background-color: #121212; color: white; font-family: 'Segoe UI';")

        self.detector = SafetyDetector()

        self.title = QLabel("SHASHTRA - A Women Protection System")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        self.title.setStyleSheet("""
            color: #FF69B4;
            padding: 20px;
            background-color: #1e1e1e;
            border-radius: 12px;
        """)

        self.image_label = QLabel("Webcam Feed")
        self.image_label.setFixedSize(700, 500)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            background-color: #2b2b2b;
            border: 3px solid #FF69B4;
            border-radius: 10px;
        """)

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

        self.btn_webcam = QPushButton("Start Webcam")
        self.btn_stop_alert = QPushButton("Stop Alert")

        self.checkbox_sos = QCheckBox("Enable SOS Detection")
        self.checkbox_lone = QCheckBox("Enable Lone Woman Detection")
        self.checkbox_sos.setChecked(True)
        self.checkbox_lone.setChecked(True)

        self.checkbox_sos.setStyleSheet("QCheckBox { font-size: 18px; padding: 10px; }")
        self.checkbox_lone.setStyleSheet("QCheckBox { font-size: 18px; padding: 10px; }")

        buttons = [self.btn_webcam, self.btn_stop_alert]
        for btn in buttons:
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
        center_layout.addWidget(self.image_label)
        center_layout.addStretch()
        main_layout.addLayout(center_layout)

        main_layout.addSpacing(10)
        main_layout.addWidget(self.result_text)

        btn_row_layout = QHBoxLayout()
        btn_row_layout.addStretch()
        btn_row_layout.addWidget(self.btn_webcam)
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

        self.btn_webcam.clicked.connect(self.start_webcam)
        self.btn_stop_alert.clicked.connect(self.detector.stop_alert)
        self.checkbox_sos.stateChanged.connect(self.toggle_sos)
        self.checkbox_lone.stateChanged.connect(self.toggle_lone_woman)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def toggle_sos(self):
        self.detector.enable_sos_detection = self.checkbox_sos.isChecked()

    def toggle_lone_woman(self):
        self.detector.enable_lone_woman_detection = self.checkbox_lone.isChecked()

    def start_webcam(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame, male_count, female_count = self.detector.analyze_frame(frame)
                image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format.Format_BGR888)
                self.image_label.setPixmap(QPixmap.fromImage(image))
                self.result_text.setPlainText(f"Males Detected: {male_count} | Females Detected: {female_count}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GenderApp()
    window.show()
    sys.exit(app.exec())
