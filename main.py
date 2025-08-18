import cv2
from ultralytics import YOLO

# Load the pre-trained YOLOv12 face detection model (adjust path if needed)
model = YOLO('yolov12n-face.pt')  # Use n for nano (fastest), or s/m/l for better accuracy

# Open the camera (0 for default webcam; change for external/ATM camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Read a frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Resize for faster processing (optional; YOLO handles various sizes)
    frame = cv2.resize(frame, (640, 480))

    # Run YOLOv12 inference on the frame
    results = model(frame, conf=0.5)  # Confidence threshold 0.5 to reduce false positives

    # Extract detections (all should be faces since it's a face-specific model)
    face_count = 0
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get coordinates and draw bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            if conf > 0.5:  # Extra filter if needed
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Face {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                face_count += 1

    # Trigger alert if 2 or more faces (intruder detected)
    if face_count >= 2:
        cv2.putText(frame, "ALERT: Intruder Detected! Multiple Faces", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print("ALERT: Intruder detected in the ATM booth! Multiple faces found.")

    # Display face count
    cv2.putText(frame, f'Faces: {face_count}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow('ATM Face Intruder Detector', frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()