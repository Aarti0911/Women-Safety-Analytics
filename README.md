SHASHTRA – Women Safety Analytics

`Shashtra.exe` is a real-time Women Safety Desktop Application developed using PyQt6, TensorFlow/Keras, MediaPipe, and YOLOv8. It provides features such as:
- Real-time gender detection using a deep learning model.
- SOS gesture recognition via webcam using MediaPipe.
- Dual camera feed support.
- Audio alerts via siren (`alert.wav`).
- Real-time object/person detection using YOLOv8.

Executable Installation

This project is distributed as a standalone `.exe` file: `Shashtra.exe`.
> Note: Ensure all dependencies are met, especially if you experience issues related to missing DLLs or incompatible hardware.

System Requirements

- Windows 10/11 (64-bit)
- Minimum 4GB RAM (8GB recommended)
- Working webcam(s) connected

Required Python Dependencies (For Source Build)

If you're running the project from source (`women_safety.py`), install the following dependencies in a Python 9.0 environment:
pip install pyqt6 opencv-python mediapipe tensorflow keras ultralytics numpy pygame datetime

File Structure (Packaged Version)

Copy
Shashtra.exe
models/
│── yolov8n.pt
│── gender_model.h5
alert.wav
README.md

How to Run?

Run Executable (No Installation Needed)
1.	Download Shashtra.exe.
2.	Place it in any folder with the following files in the same directory:
	- models/yolov8n.pt
	- models/gender_model.h5
    - alert.wav
3.	Double-click Shashtra.exe to launch the app.
If you're packaging using PyInstaller, make sure resource_path() is used to resolve file paths.

Features in Detail

Feature	                 Description
Gender Detection	     Classifies detected persons as male or female.
SOS Gesture Detection	 Detects distress gestures using MediaPipe.
Dual Camera Feed	     Monitors from two webcams.
Audio Alert	             Plays alert.wav siren when a threat is detected.

License

This project is for educational and non-commercial use only. Contact the author for permissions beyond this scope.