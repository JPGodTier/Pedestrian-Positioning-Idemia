import os

from src.ImageProcessor.ImageProcessor import ImageProcessor
from src.Common.utils import *


def main():
    # CSV Setup
    headers = ["origin", "img_id", "pedestrian_id", "bbox", "keypoints", "target"]
    threshold = 70
    # COCO Training
    coco_train_file = os.path.join(os.getcwd(),
                                   "Databases",
                                   "Coco",
                                   "annotations_trainval2017",
                                   "person_keypoints_train2017.json")
    coco_parser = ImageProcessor()
    coco_train_data = coco_parser.parse_annotation_file(coco_train_file, cat_names=["Person"], threshold=threshold,
                                                        origin="coco")

    # Saving coco train data
    save_to_csv(os.path.join(os.getcwd(), "data", "MLP_Approach", "train", f"coco_train_data{threshold}.csv"), headers,
                coco_train_data)

    # OCHumans Training
    och_train_file = os.path.join(os.getcwd(), "Databases",
                                  "OCHumans",
                                  "annotations",
                                  "ochuman_coco_format_test_range_0.00_1.00.json")

    och_parser = ImageProcessor()
    oc_train_data = och_parser.parse_annotation_file(och_train_file, cat_names=["Person"], threshold=threshold,
                                                     origin="ochumans")

    # Saving OC train data
    save_to_csv(os.path.join(os.getcwd(), "data", "MLP_Approach", "train", f"och_train_data{threshold}.csv"), headers,
                oc_train_data)

    # Saving train data
    train_data = coco_train_data + oc_train_data
    save_to_csv(os.path.join(os.getcwd(), "data", "MLP_Approach", "train", f"train_data{threshold}.csv"), headers, train_data)

    # Validation
    val_file = os.path.join(os.getcwd(),
                            "Databases",
                            "Coco",
                            "annotations_trainval2017",
                            "person_keypoints_val2017.json")
    coco_val_parser = ImageProcessor()
    val_data = coco_parser.parse_annotation_file(val_file,
                                                 cat_names=["Person"],
                                                 threshold=70, origin="coco")
    occluded_data = coco_val_parser.apply_dynamic_occlusion_to_csv(val_data)

    # Saving validation data with & without occlusion
    save_to_csv(os.path.join(os.getcwd(), "data", "MLP_Approach", "validation", "coco_validation_data.csv"),
                headers, val_data)
    save_to_csv(
        os.path.join(os.getcwd(), "data", "MLP_Approach", "validation", "coco_validation_data_with_occlusion.csv"),
        headers, occluded_data)


def visualize(train_path, val_path, occl_path):
    print("TRAINING DATA")
    print("=============")
    visualize_csv_stats(train_path)

    print("\nVALIDATION DATA")
    print("=============")
    visualize_csv_stats(val_path)

    print("\nVALIDATION DATA WITH OCCLUSION")
    print("=============")
    visualize_csv_stats(occl_path)


def startup_msg():
    print("Starting COCO Parser...")


if __name__ == "__main__":
    startup_msg()
    main()