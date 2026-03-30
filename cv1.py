# real_time_object_tracking_clean.py
import cv2
from ultralytics import YOLO

# 1. Load YOLOv8 pretrained model

model = YOLO("yolov8n.pt")  # 'n' = nano (fast for demo)

# 2. Set allowed classes 

# Uncomment or edit the list to show only specific classes
# allowed_classes = ['person', 'car']  # Example: only detect people and cars
allowed_classes = None  # Use None to allow all classes

# 3. Open webcam

cap = cv2.VideoCapture(0)  # 0 = default webcam

# 4. Detection loop

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    frame_resized = cv2.resize(frame, (640, 640))

    # Run YOLO detection with confidence threshold 0.5
    results = model(frame_resized, conf=0.5)  # Only show boxes >= 50% confident

    # Draw boxes on original frame
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()  # Bounding box coords
        scores = r.boxes.conf.cpu().numpy()  # Confidence scores
        classes = r.boxes.cls.cpu().numpy()  # Class IDs

        for box, score, cls in zip(boxes, scores, classes):
            class_name = model.names[int(cls)]

            # Skip unwanted classes if filtering is enabled
            if allowed_classes and class_name not in allowed_classes:
                continue

            x1, y1, x2, y2 = map(int, box)
            label = f"{class_name} {score:.2f}"

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("YOLOv8 Object Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5. Cleanup

cap.release()
cv2.destroyAllWindows()