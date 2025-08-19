import cv2
from ultralytics import YOLO

# Configuration Parameters (all tunable settings at the top)
MODEL_PATH = 'yolov12n-face.pt'  # YOLOv12 nano model for face detection
FRAME_WIDTH = 640  # Frame width for processing
FRAME_HEIGHT = 480  # Frame height for processing
CONF_THRESHOLD = 0.6  # Confidence threshold for detections (higher reduces false positives)
IOU_THRESHOLD = 0.45  # IoU for non ˓˓-max suppression (lower merges more overlapping boxes)
MIN_FACE_SIZE = 50  # New: Minimum face width/height in pixels to filter small detections
MAX_FACES = 2  # New: Maximum faces to consider (caps noise in crowded scenes)
ALERT_THRESHOLD = 3  # Number of consecutive frames with 2+ faces for alert
CLIP_LIMIT = 2.0  # New: CLAHE clip limit for adaptive contrast adjustment
CAMERA_INDEX = 0  # Camera index (0 for default webcam)

# Load the YOLOv12 model
model = YOLO(MODEL_PATH)

# Initialize CLAHE for adaptive contrast (new: replaces equalizeHist)
clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=(8, 8))

# Open the camera
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize temporal smoothing counter
alert_counter = 0

while True:
    # Read a frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Resize frame
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # Preprocess: Apply CLAHE for better lighting handling (new)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = clahe.apply(frame_gray)
    frame_processed = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

    # Run YOLOv12 inference with tuned parameters
    results = model(frame_processed, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

    # Extract detections
    face_count = 0
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            # New: Filter small detections
            if conf > CONF_THRESHOLD and (x2 - x1) > MIN_FACE_SIZE and (y2 - y1) > MIN_FACE_SIZE:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Face {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                face_count += 1

    # New: Cap face count to avoid noise from excessive detections
    face_count = min(face_count, MAX_FACES)

    # Temporal smoothing for intruder alert
    if face_count >= 2:
        alert_counter += 1
        if alert_counter >= ALERT_THRESHOLD:
            cv2.putText(frame, "ALERT: Intruder Detected! Multiple Faces", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print("ALERT: Intruder detected in the ATM booth! Multiple faces found.")
    else:
        alert_counter = 0

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