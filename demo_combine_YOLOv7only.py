import argparse
import logging
import time
from typing import Optional, Tuple

import albumentations as A
import cv2
import mediapipe as mp
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

import onnxruntime

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

from constants import targets
from custom_utils.utils import build_model

logging.basicConfig(format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s", level=logging.INFO)

COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

class Demo:
    @staticmethod
    def preprocess(img: np.ndarray, input_size: Tuple[int, int]) -> np.ndarray:
        img = cv2.resize(img, input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1).astype(np.float32)
        img /= 255.0
        return img

    @staticmethod
    def preprocess2(img: np.ndarray, transform_classify) -> Tensor:
        transformed_image = transform_classify(image=img)
        return transformed_image["image"]

    @staticmethod
    def get_transform_classification_for_inf(transform_config: DictConfig):
        transforms_list = [getattr(A, key)(**params) for key, params in transform_config.items()]
        transforms_list.append(ToTensorV2())
        return A.Compose(transforms_list)

    @staticmethod
    def run(
        detector, classifier, transform_classify, conf: DictConfig, num_hands: int = 2, threshold: float = 0.5, landmarks: bool = False
    ) -> None:
        if landmarks:
            hands = mp.solutions.hands.Hands(
                model_complexity=0, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8
            )

        cap = cv2.VideoCapture(0)

        # Get input details of the YOLO model
        input_name = detector.get_inputs()[0].name
        input_shape = detector.get_inputs()[0].shape
        input_width, input_height = input_shape[2], input_shape[3]

        t1 = cnt = 0
        while cap.isOpened():
            delta = time.time() - t1
            t1 = time.time()

            ret, frame = cap.read()
            if ret:
                # Preprocess for YOLO
                processed_image = Demo.preprocess(frame, (input_width, input_height))

                # Run YOLO inference
                ort_inputs = {input_name: processed_image[np.newaxis, ...]}
                outputs = detector.run(None, ort_inputs)

                # Process YOLO output
                if len(outputs) > 0 and outputs[0] is not None and len(outputs[0]) > 0:
                    output = outputs[0][0]  # Assuming batch size 1
                    boxes = output[:, :4]  # First 4 columns are box coordinates
                    scores = output[:, 4]  # 5th column is objectness score
                    class_ids = output[:, 5]  # 6th column onwards are class probabilities

                    # Apply NMS
                    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), threshold, 0.4)
                    
                    for i in indices:
                        box = boxes[i]
                        score = scores[i]
                        class_id = int(class_ids[i])

                        x1, y1, x2, y2 = box
                        x1 = int(x1 * frame.shape[1] / input_width)
                        y1 = int(y1 * frame.shape[0] / input_height)
                        x2 = int(x2 * frame.shape[1] / input_width)
                        y2 = int(y2 * frame.shape[0] / input_height)

                        cutting_frame = frame[y1:y2, x1:x2]
                        if cutting_frame.size > 0:
                            # Process cutting frame for classification
                            processed_frame = Demo.preprocess2(cutting_frame, transform_classify)
                            with torch.no_grad():
                                output = classifier([processed_frame])
                            label = output["labels"].argmax(dim=1)

                            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, thickness=3)
                            cv2.putText(frame, f"{targets[class_id]}: {targets[int(label) + 1]}", (x1, y1 - 10), FONT, 0.5, COLOR, 2)

                if landmarks:
                    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                mp.solutions.hands.HAND_CONNECTIONS,
                                mp_drawing_styles.DrawingSpec(color=[0, 255, 0], thickness=2, circle_radius=1),
                                mp_drawing_styles.DrawingSpec(color=[255, 255, 255], thickness=1, circle_radius=1),
                            )

                fps = 1 / delta
                cv2.putText(frame, f"FPS: {fps:.2f}, Frame: {cnt}", (30, 30), FONT, 1, COLOR, 2)
                cnt += 1

                cv2.imshow("Frame", frame)

                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()

def parse_arguments(params: Optional[Tuple] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo detection...")
    parser.add_argument("-pd", "--path_to_config_detection", required=False, type=str, default="configs/RetinaNet_ResNet50.yaml", help="Path to detection config")
    parser.add_argument("-pc", "--path_to_config_classification", required=False, type=str, default="configs/ResNeXt50.yaml", help="Path to classification config")
    parser.add_argument("-lm", "--landmarks", required=False, action="store_true", help="Use landmarks")

    known_args, _ = parser.parse_known_args(params)
    return known_args

if __name__ == "__main__":
    args = parse_arguments()
    args.landmarks = True

    model_detection = onnxruntime.InferenceSession("YoloV7Tiny.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    args.path_to_config_classification = "configs/ResNet18.yaml"
    conf2 = OmegaConf.load(args.path_to_config_classification)
    model_classification = build_model(conf2)
    transform2 = Demo.get_transform_classification_for_inf(conf2.test_transforms)
    
    model_classification.eval()

    if model_detection is not None:
        Demo.run(model_detection, model_classification, transform2, conf2, num_hands=100, threshold=0.25, landmarks=args.landmarks)