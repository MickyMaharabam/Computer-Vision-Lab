import cv2
from ultralytics import YOLO

# Load models
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
yolo_model = YOLO('yolov8n.pt')

# Open webcam
cap = cv2.VideoCapture(0)

# Start with Haar Cascade
mode = 'haar'

print("Press 'h' for Haar, 'y' for YOLO, 'ESC' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if mode == 'haar':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Haar Cascade Mode", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    elif mode == 'yolo':
        results = yolo_model(frame)
        frame = results[0].plot()
        cv2.putText(frame, "YOLO Mode", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Real-time Object Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('h'):
        mode = 'haar'
    elif key == ord('y'):
        mode = 'yolo'

cap.release()
cv2.destroyAllWindows()
