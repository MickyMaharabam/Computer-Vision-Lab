import cv2
from ultralytics import YOLO

# Load YOLOv8 model
yolo_model = YOLO('yolov8n.pt')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = yolo_model(frame)
    yolo_frame = results[0].plot()  # Annotated frame

    cv2.imshow('YOLOv8 Real-time Detection', yolo_frame)

    # Press ESC key to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
