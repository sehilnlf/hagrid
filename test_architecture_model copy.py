import cv2
import numpy as np
import onnxruntime
import time
from constants import targets

# Constants
CONFIDENCE_THRESHOLD = 0.1  # Lowered threshold for testing
NMS_THRESHOLD = 0.4
COLOR = (0, 255, 0)  # Green color for bounding boxes
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize ONNX Runtime session
detector = onnxruntime.InferenceSession("YoloV7Tiny.onnx", providers=['CPUExecutionProvider'])

# Get model input details
model_inputs = detector.get_inputs()
input_name = model_inputs[0].name
input_shape = model_inputs[0].shape
# print(f"Model input name: {input_name}, shape: {input_shape}")

# Get model output details
model_outputs = detector.get_outputs()
# for output in model_outputs:
#     print(f"Model output name: {output.name}, shape: {output.shape}")

t1 = cnt = 0

while cap.isOpened():
    delta = time.time() - t1
    t1 = time.time()

    ret, frame = cap.read()
    if ret:
        # Preprocess the image
        input_height, input_width = input_shape[2], input_shape[3]
        frame_resized = cv2.resize(frame, (input_width, input_height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        input_data = np.transpose(frame_rgb, (2, 0, 1)).astype(np.float32)
        input_data = input_data / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        
        # Run inference
        outputs = detector.run(None, {input_name: input_data})

        # Process output
        if len(outputs) > 0 and outputs[0].size > 0:
            output = outputs[0]
            # print(outputs)
            # break
            # Assuming YOLO v7 output format: [x, y, w, h, confidence, class_prob1, class_prob2, ...]
            boxes = output[:, 1:5]
            scores = output[:, -1]
            class_ids = output[:, 5]
            # print(boxes)
            
            # Apply NMS
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

            for i in indices:
                box = boxes[i]
                score = scores[i]
                class_id = class_ids[i]

                # x, y, w, h = box
                
                x1, y1, x2, y2 = box
                x1 = int(x1 / 224 * frame.shape[1])
                y1 = int(y1 / 224 * frame.shape[0])
                x2 = int(x2 / 224 * frame.shape[1])
                y2 = int(y2 / 224 * frame.shape[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, thickness=2)
                label = f"Class: {targets[class_id + 1]}, Conf: {score:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), FONT, 0.5, COLOR, 1)

        else:
            # print("No detections in this frame")
            pass

        # Display FPS
        fps = 1 / delta
        cv2.putText(frame, f"FPS: {fps:.2f}, Frame: {cnt}", (10, 30), FONT, 1, COLOR, 2)
        cnt += 1

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        print("time:", time.time() - t1)
        # break
    else:
        # break

        cap.release()
        cv2.destroyAllWindows()

