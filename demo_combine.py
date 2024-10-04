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

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

from constants import targets
from custom_utils.utils import build_model

logging.basicConfig(format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s", level=logging.INFO)

COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX



class Demo:
    @staticmethod
    def preprocess(img: np.ndarray, transform_detect) -> Tuple[Tensor, Tuple[int, int], Tuple[int, int]]:
        """
        Preproc image for model input
        Parameters
        ----------
        img: np.ndarray
            input image
        transform :
            albumentation transforms
        """
        height, width = img.shape[0], img.shape[1]
        # print(img)
        transformed_image = transform_detect(image=img)
        processed_image = transformed_image["image"] / 255.0
        return processed_image, (width, height)
    
    @staticmethod
    def preprocess2(img: np.ndarray, transform_classify) -> Tuple[Tensor, Tuple[int, int], Tuple[int, int]]:
        """
        Preproc image for model input
        Parameters
        ----------
        img: np.ndarray
            input image
        transform :
            albumentation transforms
        """
        transformed_image = transform_classify(image=img)
        return transformed_image["image"]

    @staticmethod
    def get_transform_classification_for_inf(transform_config: DictConfig):
        """
        Create list of transforms from config
        Parameters
        ----------
        transform_config: DictConfig
            config with test transforms
        """
        transforms_list = [getattr(A, key)(**params) for key, params in transform_config.items()]
        transforms_list.append(ToTensorV2())
        return A.Compose(transforms_list)
    
    @staticmethod
    def get_transform_detection_for_inf(transform_config: DictConfig):
        """
        Create list of transforms from config
        Parameters
        ----------
        transform_config: DictConfig
            config with test transforms
        """
        transforms_list = [getattr(A, key)(**params) for key, params in transform_config.items()]
        transforms_list.append(ToTensorV2())
        return A.Compose(transforms_list)

    @staticmethod
    def run(
        detector, classifier, transform_detect, transform_classify, conf: DictConfig, num_hands: int = 2, threshold: float = 0.5, landmarks: bool = False
    ) -> None:
        """
        Run detection model and draw bounding boxes on frame
        Parameters
        ----------
        detector : TorchVisionModel
            Detection model
        transform :
            albumentation transforms
        transform_config: DictConfig
            config with test transforms
        num_hands:
            Min hands to detect
        threshold : float
            Confidence threshold
        landmarks : bool
            Detect landmarks
        """

        if landmarks:
            hands = mp.solutions.hands.Hands(
                model_complexity=0, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8
            )

        cap = cv2.VideoCapture(0)

        t1 = cnt = 0
        while cap.isOpened():
            delta = time.time() - t1
            t1 = time.time()

            ret, frame = cap.read()
            if ret:
                processed_image, size = Demo.preprocess(frame, transform_detect)
                with torch.no_grad():
                    output = detector([processed_image])[0]

                boxes = output["boxes"][:num_hands]
                scores = output["scores"][:num_hands]
                labels = output["labels"][:num_hands]

                if landmarks:
                    results = hands.process(frame[:, :, ::-1])
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                mp.solutions.hands.HAND_CONNECTIONS,
                                mp_drawing_styles.DrawingSpec(color=[0, 255, 0], thickness=2, circle_radius=1),
                                mp_drawing_styles.DrawingSpec(color=[255, 255, 255], thickness=1, circle_radius=1),
                            )
                for i in range(min(num_hands, len(boxes))):
                    if scores[i] > threshold:
                        width, height = size
                        scale = max(width, height) / conf.LongestMaxSize.max_size
                        padding_w = abs(conf.PadIfNeeded.min_width - width // scale) // 2
                        padding_h = abs(conf.PadIfNeeded.min_height - height // scale) // 2

                        x1 = int((boxes[i][0].item() - padding_w) * scale)
                        y1 = int((boxes[i][1] - padding_h) * scale)
                        x2 = int((boxes[i][2] - padding_w) * scale)
                        y2 = int((boxes[i][3] - padding_h) * scale)
                        print(frame.shape)

                        cutting_frame = np.array(frame[y1:y2, x1:x2, :])
                        print(cutting_frame.shape)
                        processed_frame = Demo.preprocess2(cutting_frame, transform_classify)
                        with torch.no_grad():
                            output = classifier([processed_frame])
                        print(output)
                        label = output["labels"].argmax(dim=1)


                        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, thickness=3)
                        #####                         cv2.putText

                        cv2.putText(
                            frame, targets[int(label) + 1], (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=3
                        )

                fps = 1 / delta
                cv2.putText(frame, f"FPS: {fps :02.1f}, Frame: {cnt}", (30, 30), FONT, 1, COLOR, 2)
                cnt += 1

                cv2.imshow("Frame", frame)

                key = cv2.waitKey(1)
                if key == ord("q"):
                    return
            else:
                cap.release()
                cv2.destroyAllWindows()


def parse_arguments(params: Optional[Tuple] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo detection...")

    # parser.add_argument("-p", "--path_to_config", required=True, type=str, help="Path to config")
    parser.add_argument("-pd", "--path_to_config_detection", required=False, type=str, default="configs/RetinaNet_ResNet50.yaml", help="Path to detection config")
    parser.add_argument("-pc", "--path_to_config_classification", required=False, type=str, default="configs/ResNeXt50.yaml", help="Path to classification config")

    parser.add_argument("-lm", "--landmarks", required=False, action="store_true", help="Use landmarks")

    known_args, _ = parser.parse_known_args(params)
    return known_args


if __name__ == "__main__":
    args = parse_arguments()

    args.path_to_config_detection = "configs/RetinaNet_ResNet50.yaml"
    args.path_to_config_classification = "configs/ResNet18.yaml"
    
    args.landmarks = True

    # config
    conf1 = OmegaConf.load(args.path_to_config_detection) # OmegaConf ~ to load yaml file
    conf2 = OmegaConf.load(args.path_to_config_classification)
    
    model_detection = build_model(conf1)
    transform1 = Demo.get_transform_detection_for_inf(conf1.test_transforms)
    if conf1.model.checkpoint is not None:
        snapshot = torch.load(conf1.model.checkpoint, map_location=torch.device("cpu"))
        model_detection.load_state_dict(snapshot["MODEL_STATE"])

    model_classification = build_model(conf2)
    transform2 = Demo.get_transform_classification_for_inf(conf2.test_transforms)
    if conf2.model.checkpoint is not None:
        snapshot = torch.load(conf2.model.checkpoint, map_location=torch.device("cpu"))
        model_classification.load_state_dict(snapshot["MODEL_STATE"])
    
    model_detection.eval()
    model_classification.eval()

    if model_detection is not None:
        Demo.run(model_detection, model_classification, transform1, transform2, conf1.test_transforms, num_hands=100, threshold=0.8, landmarks=args.landmarks)
