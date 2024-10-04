import cv2
import time
from ultralytics import YOLO

# Constants
CONFIDENCE_THRESHOLD = 0.1  # Lowered threshold for testing
NMS_THRESHOLD = 0.4
COLOR = (0, 255, 0)  # Green color for bounding boxes
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize YOLO model
detector = YOLO("D:/Empty/hagrid/YOLOv10x_hands.pt")

detector.export



t1 = cnt = 0

while cap.isOpened():
    delta = time.time() - t1
    t1 = time.time()

    ret, frame = cap.read()
    if ret:
        # Run YOLO inference
        outputs = detector(frame)
        
        # Check if any detections are made
        if len(outputs) > 0:
            output = outputs[0]
            if len(output.boxes) > 0:
                boxes = output.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format
                scores = output.boxes.conf.cpu().numpy()
                class_ids = output.boxes.cls.cpu().numpy()

                # Apply NMS
                indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

                for i in indices:
                    box = boxes[i]
                    score = scores[i]
                    class_id = int(class_ids[i])

                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, thickness=2)
                    label = f"Class: {outputs[0].names[class_id]}, Conf: {score:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), FONT, 0.5, COLOR, 1)

        # Display FPS
        fps = 1 / delta
        cv2.putText(frame, f"FPS: {fps:.2f}, Frame: {cnt}", (10, 30), FONT, 1, COLOR, 2)
        cnt += 1

        # cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print("Time", time.time() - t1)
    else:
        cap.release()
        cv2.destroyAllWindows()
