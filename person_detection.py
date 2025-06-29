from ultralytics import YOLO
import cv2

# Load YOLOv8 model (automatically downloads if not present)
model = YOLO('yolov8n.pt')  # 'n' for nano (smallest), can use 's', 'm', 'l', 'x' for larger models

# Open video source (0 for webcam, or file path)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame, classes=0)  # Class 0 is 'person' in COCO dataset

    # Visualize results
    annotated_frame = results[0].plot()  # Draw bounding boxes and labels

    # Display frame
    cv2.imshow("Person Detection", annotated_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
