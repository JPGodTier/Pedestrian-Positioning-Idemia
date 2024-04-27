import os
import torch
import pandas as pd
from mmdeploy_runtime import PoseDetector
import cv2
import numpy as np
import yaml
import time

from src.ImageParser.ImageProcessor import ImageProcessor
from src.Common.utils import save_to_csv


def load_config(config_path):
    with open(config_path, 'r') as file:
        p_config = yaml.safe_load(file)
    return p_config


def zero_out_low_confidence_keypoints(keypoints, threshold=0.5):
    for i in range(2, len(keypoints), 3):
        if keypoints[i] < threshold:
            keypoints[i-2] = 0  # Zero out x
            keypoints[i-1] = 0  # Zero out y
            keypoints[i] = 0  # Zero out confidence
    return keypoints


def main(device_name, csv_file):
    model_path = "bin\\rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.zip"
    detector = PoseDetector(model_path=model_path, device_name="cpu", device_id=0)

    df = pd.read_csv(csv_file)
    results = []
    count = 0
    for index, row in df.iterrows():
        img_id = row['img_id']
        img_filename = f"{str(img_id).zfill(12)}.jpg"
        img_path = os.path.join(os.getcwd(), "Coco", "train2017", "train2017", img_filename)

        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        bbox = eval(row['bbox']) if isinstance(row['bbox'], str) else None

        new_bbox = np.array(bbox, dtype=int)
        new_bbox[2:] += new_bbox[:2]
        result = detector(img, new_bbox)
        result = result.reshape(-1).tolist()
        if np.isnan(result).any():
            count += 1
            # print(f"Keypoints not generated for image: {img_filename}")
            continue

        # updated_keypoints = zero_out_low_confidence_keypoints(result, 0.5)

        normalized_bbox = bbox.copy()
        normalized_bbox[:2] = [0, 0]
        # result, _ = ImageProcessor.normalize_keypoints(updated_keypoints[:-6], normalized_bbox)
        result, _ = ImageProcessor.normalize_keypoints(result[:-6], normalized_bbox)

        results.append([img_id, row['pedestrian_id'], row['bbox'], result, row['target']])

    headers = ["img_id", "pedestrian_id", "bbox", "keypoints", "target"]
    save_to_csv("data\\train_data_RTMpose.csv", headers, results)
    print(f"Processed {len(results)} images and saved to train_data_RTMpose.csv\nFailed images : {count}")


if __name__ == '__main__':
    config = load_config(os.path.join(os.getcwd(), "config", "config.yaml"))
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print("Running on device: ", device)

    start_time = time.time()

    main(device, "data\\coco_train_data.csv")

    end_time = time.time()
    print(f"Execution Time: {end_time - start_time:.2f} seconds")
