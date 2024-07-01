import cv2
import torch
import numpy as np
import os

from torchvision.transforms import functional
from src.Calibration.Calibration import Calibration
from src.Pipelines.ModelLoader import load_fasterrcnn, load_pose_detector
from src.Common.utils import load_mlp_model


class Pipeline3DProcessor:
    def __init__(self, device, cnn_model_path, mlp_model_path, calibration_path, mlp_settings):
        self.device = device
        self.model_detector = load_pose_detector(cnn_model_path, device_name="cpu", device_id=0)
        self.mlp_model = self.load_mlp_model(mlp_model_path, mlp_settings)
        self.calibration = Calibration(calibration_path)
        self.cnn_model = load_fasterrcnn(device)
        self.cnn_model.eval()
        self.cnn_model.to(device)
        self.mlp_model.eval()
        self.mlp_model.to(device)

    # -----------------------------------------------------------------------------
    # load_mlp_model
    # -----------------------------------------------------------------------------
    def load_mlp_model(self, model_path, mlp_settings):
        model = load_mlp_model(self.device,
                               model_path,
                               mlp_settings["input_size"],
                               mlp_settings["output_size"],
                               mlp_settings["layers"])
        return model

    # -----------------------------------------------------------------------------
    # normalize_keypoints
    # -----------------------------------------------------------------------------
    @staticmethod
    def normalize_keypoints(keypoints, bbox):
        normalized_keypoints = []
        bbox_x, bbox_y, bbox_width, bbox_height = bbox
        for i in range(0, len(keypoints), 3):
            norm_x = (keypoints[i] - bbox_x) / bbox_width
            norm_y = (keypoints[i + 1] - bbox_y) / bbox_height
            conf = keypoints[i + 2]
            normalized_keypoints.extend([norm_x, norm_y, conf])
        return normalized_keypoints

    # -----------------------------------------------------------------------------
    # zero_out_low_confidence_keypoints
    # -----------------------------------------------------------------------------
    @staticmethod
    def zero_out_low_confidence_keypoints(keypoints, threshold=0.5):
        """Set keypoints confidence to zero if below a given threshold."""
        for i in range(2, len(keypoints), 3):
            if keypoints[i] < threshold:
                keypoints[i] = 0  # Zero out confidence
        return keypoints

    # -----------------------------------------------------------------------------
    # infer_2d_position
    # -----------------------------------------------------------------------------
    def infer_2d_position(self, keypoints):
        input_tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.mlp_model(input_tensor).squeeze().tolist()
        return prediction

    # -----------------------------------------------------------------------------
    # rtm_infer
    # -----------------------------------------------------------------------------
    def rtm_infer(self, frame, bbox):
        try:
            bbox = np.array(bbox, dtype=int)
            bbox[2:] += bbox[:2]

            inference_result = self.model_detector(frame, bbox)
            # keypoints = inference_result[0, :-2, :].reshape(-1).tolist()
            keypoints = inference_result[0, :, :].reshape(-1).tolist()

            # Zeroise low confidence kps
            updated_kps = self.zero_out_low_confidence_keypoints(keypoints)

            return updated_kps
        except Exception as e:
            print(f"Inference failed: {str(e)}")
            return None

    # -----------------------------------------------------------------------------
    # detect_people
    # -----------------------------------------------------------------------------
    def detect_people(self, frame, threshold=0.9):
        frame_tensor = functional.to_tensor(frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            predictions = self.cnn_model(frame_tensor)
        pred_classes = predictions[0]['labels']
        pred_scores = predictions[0]['scores']
        pred_boxes = predictions[0]['boxes']

        # Filter detections based on the class label and score threshold
        filtered_indices = [
            i for i, (cls, score) in enumerate(zip(pred_classes, pred_scores))
            if cls == 1 and score > threshold
        ]
        high_conf_boxes = pred_boxes[filtered_indices]
        high_conf_scores = pred_scores[filtered_indices]

        # Calculate the center of the frame
        frame_center = torch.tensor([frame.shape[1] / 2, frame.shape[0] / 2])

        # Select the most centrally located box
        if len(high_conf_boxes) > 0:
            main_index = min(range(len(high_conf_boxes)), key=lambda i: torch.norm(torch.tensor([
                (high_conf_boxes[i][0] + high_conf_boxes[i][2]) / 2,
                (high_conf_boxes[i][1] + high_conf_boxes[i][3]) / 2
            ]) - frame_center))
            main_box = high_conf_boxes[main_index]
            main_score = high_conf_scores[main_index]
        else:
            return None, None, []

        # Calculate the bounding box dimensions
        x_min, y_min, x_max, y_max = map(int, main_box.tolist())
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        inferred_keypoints = self.rtm_infer(frame, bbox)
        normalized_keypoints = self.normalize_keypoints(inferred_keypoints, bbox)

        return main_box.cpu(), main_score.cpu(), (normalized_keypoints, bbox)