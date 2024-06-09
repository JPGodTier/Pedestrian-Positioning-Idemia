import os
import torch
import pandas as pd
from mmdeploy_runtime import PoseDetector
import cv2
import numpy as np
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import ast

from src.ImageProcessor.ImageProcessor import ImageProcessor
from src.Common.utils import save_to_csv, load_config


def parse_bbox(bbox_str):
    try:
        return ast.literal_eval(bbox_str)
    except ValueError:
        return None


def zero_out_low_confidence_keypoints(keypoints, threshold=0.5):
    """Set keypoints confidence to zero if below a given threshold."""
    for i in range(2, len(keypoints), 3):
        if keypoints[i] < threshold:
            keypoints[i] = 0  # Zero out confidence
    return keypoints


def rtm_infer(detector, img_path, bbox: list):
    """
    Infer the pose of the pedestrian using the CNN detector.

    Args:
        detector: CNN detector initialized and ready to infer.
        img_path: Path to the image to be processed.
        bbox: List representing the bounding box [x, y, width, height].

    Returns:
        List of inferred keypoints, or None if inference fails.
    """
    if not os.path.exists(img_path):
        logging.error(f"Image not found: {img_path}")
        return None

    img = cv2.imread(img_path)
    if img is None:
        logging.error(f"Failed to load image at {img_path}")
        return None

    try:
        # Prepare bbox for inference.
        bbox = np.array(bbox, dtype=int)
        bbox[2:] += bbox[:2]  # Convert from [x, y, w, h] to [x1, y1, x2, y2]
        inference_result = detector(img, bbox).reshape(-1).tolist()
        return inference_result
    except Exception as e:
        logging.error(f"Inference failed for image {img_path}: {str(e)}")
        return None


def image_finder(origin: str, img_id: str, images_paths):
    """ Find the image path based on its id & origin

    Args:
        origin: image origin (can be Coco, OCHumans or any other database)
        img_id: image number id
        images_paths: paths to downloaded database

    Returns:
        Full path to given image
    """
    img_path = ""

    # Process image based on their origin
    if origin == "coco":
        img_filename = f"{img_id.zfill(12)}.jpg"
        img_path = os.path.join(images_paths[origin], img_filename)
    elif origin == "ochumans":
        img_filename = f"{img_id.zfill(6)}.jpg"
        img_path = os.path.join(images_paths[origin], img_filename)

    return img_path


def process_image(detector, images_paths, row, confidence_threshold):
    """
    Processes a single image: finds the image, runs inference, and normalizes keypoints.

    Args:
        row (Series): DataFrame row containing image and bbox information.
        detector (PoseDetector): The detector to use for running inference.
        images_paths (dict): Dictionary of image paths by origin.
        confidence_threshold (float): Confidence threshold for zeroing out low-confidence keypoints.

    Returns:
        tuple: A tuple containing the processed data or None if an error occurs, and an error flag.
    """
    img_path = image_finder(row[1], row[2], images_paths)
    if not os.path.exists(img_path):
        logging.error(f"Image not found: {img_path}")
        return None, True

    bbox = row[4]
    if not bbox:
        logging.error(f"Bbox error for image: {img_path}")
        return None, True

    inferred_keypoints = rtm_infer(detector, img_path, bbox)
    if inferred_keypoints is None or np.isnan(inferred_keypoints).any():
        return None, True

    if confidence_threshold == -1:
        updated_keypoints = inferred_keypoints
    else:
        updated_keypoints = zero_out_low_confidence_keypoints(inferred_keypoints, confidence_threshold)

    # Normalize inferred keypoints
    normalized_bbox = bbox.copy()
    normalized_bbox[:2] = [0, 0]
    result, _ = ImageProcessor.normalize_keypoints(updated_keypoints, normalized_bbox)

    result_list = [row[1], row[2], row[3], row[4], result, row[6]]
    return result_list, False


def create_rtm_csv_parallel(detector, images_paths, input_csv, confidence_threshold=0.5):
    # Column data types
    dtypes = {
        'origin': 'category',
        'img_id': str,
        'pedestrian_id': int,
        'bbox': str,
        'target': str
    }

    # Reading CSV
    df = pd.read_csv(input_csv, dtype=dtypes)
    df['bbox'] = df['bbox'].apply(parse_bbox)

    # Return Values init
    results = []
    error_count = 0

    # Use ThreadPoolExecutor to parallelize image processing
    with ThreadPoolExecutor(max_workers=None) as executor:
        futures = [executor.submit(process_image, detector, images_paths, row, confidence_threshold)
                   for row in df.itertuples()]
        for future in futures:
            result, error = future.result()
            if error:
                error_count += 1
            elif result:
                results.append(result)

    return results, error_count


def main(csv_train_file, csv_val_file, cnn_path, train_out_path, val_out_path):
    try:
        model_path = cnn_path
        model_detector = PoseDetector(model_path=model_path, device_name="cpu", device_id=0)
    except RuntimeError as e:
        raise Exception(f"Error while loading CNN model: {e}")

    headers = ["origin", "img_id", "pedestrian_id", "bbox", "keypoints", "target"]

    # Images Paths
    coco_train_path = os.path.join(os.getcwd(), "Databases", "Coco", "train2017", "train2017")
    coco_val_path = os.path.join(os.getcwd(), "Databases","Coco", "val2017")
    och_train_path = os.path.join(os.getcwd(), "Databases","OCHumans", "images", "images")
    train_image_paths = {"coco": coco_train_path, "ochumans": och_train_path}
    val_image_paths = {"coco": coco_val_path}

    # RTM Train data creation
    rtm_train_data, error_count = create_rtm_csv_parallel(model_detector, train_image_paths, csv_train_file)
    save_to_csv(train_out_path, headers, rtm_train_data)
    print(f"Processed {len(rtm_train_data)} images and saved to train_data_RTMpose.csv\nFailed images : {error_count}")

    # RTM Val data creation
    rtm_val_data, error_count = create_rtm_csv_parallel(model_detector, val_image_paths, csv_val_file)
    coco_val_parser = ImageProcessor()
    occluded_data = coco_val_parser.apply_dynamic_occlusion_to_csv(rtm_val_data)
    save_to_csv(val_out_path, headers, occluded_data)
    print(f"Processed {len(rtm_val_data)} images and saved to val_data_RTMpose.csv\nFailed images : {error_count}")


if __name__ == '__main__':
    # Load project config
    config = load_config(os.path.join(os.getcwd(), "config", "config.yaml"))
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    cnn_model_path = config["cnn"]["model_path"]
    print(f"Running on device: {device}")

    # Time execution
    start_time = time.time()

    # Configure train & val data (based on MLP train & validation data)
    train_data = "data\\MLP_Approach\\train\\train_data.csv"
    validation_data = "data\\MLP_Approach\\validation\\coco_validation_data.csv"

    # Configure output paths
    train_output_path = "data\\RTM_Approach\\train\\train_data_RTMpose.csv"
    val_output_path = "data\\RTM_Approach\\validation\\val_data_RTMpose.csv"
    main(train_data, validation_data, cnn_model_path, train_output_path, val_output_path)

    end_time = time.time()
    print(f"Execution Time: {end_time - start_time:.2f} seconds")
