import logging
import os
import random
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import skimage.io as io
import numpy as np
import torch
import pandas as pd


# -----------------------------------------------------------------------------
# ImageProcessor
# -----------------------------------------------------------------------------
class ImageProcessor:
    """ Initializes an Image Processor Object

    """

    def __init__(self):
        pass

    # -----------------------------------------------------------------------------
    # __kps_visibility_check
    # -----------------------------------------------------------------------------
    @staticmethod
    def __kps_visibility_check(keypoint_list, threshold) -> bool:
        """ Checks keypoints visibility against given threshold.

        Args:
            keypoint_list: Coco keypoint list
            threshold: Required % of visible keypoints

        Returns:
            True if threshold is met, False otherwise
        """
        # Extracting visibility values from kps
        v_values = keypoint_list[2::3]

        # Counting number of non-zero visibility values
        visible_count = sum(v != 0 for v in v_values)

        # Checking visibility count against threshold
        total_kps = len(v_values)
        visible_percentage = (visible_count / total_kps) * 100

        return visible_percentage >= threshold

    # -----------------------------------------------------------------------------
    # normalize_keypoints
    # -----------------------------------------------------------------------------
    @staticmethod
    def normalize_keypoints(keypoints, bbox):
        """ Normalizes keypoints to be relative to the bounding box.

        Args:
            keypoints: List of keypoints for an image.
            bbox: Bounding box with [x0, y0, width, height].

        Returns:
            List of normalized keypoints, and normalized bbox
        """
        x0, y0, width, height = bbox
        if width == 0 or height == 0:
            raise Exception(f"Error while normalizing: width={width} | height={height}")

        norm_keypoints = []
        for i in range(0, len(keypoints), 3):
            x, y, v = keypoints[i:i+3]
            if v != 0:
                x_norm = (x - x0) / width
                y_norm = (y - y0) / height
                norm_keypoints.extend([x_norm, y_norm, v])
            else:
                norm_keypoints.extend([x, y, v])

        return norm_keypoints, bbox

    @staticmethod
    def apply_dynamic_occlusion_to_csv(parsed_data, **kwargs):
        """
        Applies dynamic occlusion to a list of parsed data points and returns the modified data.

        This method randomly selects one of three occlusion types ('no_occlusion', 'box', 'keypoints')
        and applies the corresponding occlusion to each data point in the input list.

        Args:
            parsed_data (list): A list of data points where each data point is expected to be a list
                containing [origin, img_id, annotation_id, bbox, keypoints, target].
            **kwargs: Additional keyword arguments for controlling the occlusion behavior.
                - box_occlusion_chance (float): Probability of applying box occlusion (default is 0.7).
                - box_scale_factor (tuple): Scale factor range for box occlusion (default is (0.5, 1)).
                - kp_weight_position (str): Weight position for keypoints occlusion, e.g., "upper_body" (default is "upper_body").
                - kp_weight_value (float): Weight value for keypoints occlusion (default is 0.7).
                - kp_min_threshold (int): Minimum number of visible keypoints for keypoints occlusion (default is 5).

        Returns:
            list: A list of modified data points with applied occlusion. Each data point in the list is a
                list containing [origin, img_id, annotation_id, modified_bbox, modified_keypoints, modified_target].

        """
        modified_data = []
        for data_point in parsed_data:
            origin, img_id, annotation_id, bbox, keypoints, target = data_point
            occlusion_type = random.choice(['no_occlusion', 'box', 'keypoints'])

            if occlusion_type == 'box':
                occlusion_chance = kwargs.get('box_occlusion_chance', 0.7)
                box_scale_factor = kwargs.get('box_scale_factor', (0.5, 1))
                occluded_bbox, occluded_keypoints, occluded_target = ImageProcessor.apply_box_occlusion(keypoints,
                                                                                                        bbox,
                                                                                                        target,
                                                                                                        occlusion_chance,
                                                                                                        box_scale_factor)
                modified_data.append([origin, img_id, annotation_id, occluded_bbox, occluded_keypoints, occluded_target])
            elif occlusion_type == 'keypoints':
                weight_position = kwargs.get('kp_weight_position', "upper_body")
                weight_value = kwargs.get('kp_weight_value', 0.7)
                min_visible_threshold = kwargs.get('kp_min_threshold', 5)
                occluded_keypoints = ImageProcessor.apply_keypoints_occlusion(keypoints,
                                                                              weight_position,
                                                                              weight_value,
                                                                              min_visible_threshold)
                modified_data.append([origin, img_id, annotation_id, bbox, occluded_keypoints, target])
            elif occlusion_type == "no_occlusion":
                modified_data.append([origin, img_id, annotation_id, bbox, keypoints, target])
                pass

        return modified_data

    # -----------------------------------------------------------------------------
    # __parse_images
    # -----------------------------------------------------------------------------
    def __parse_images(self, coco_db, image_ids, threshold, origin):
        """ Parsing Images following project criteria.

        Args:
            image_ids: Selected Images
            threshold: Required % of visible keypoints

        Returns:
            List of data.
        """
        parsed_data = []

        # Iterating through images
        for img_id in image_ids:
            img_anns = coco_db.loadAnns(
                coco_db.getAnnIds(img_id))

            # Iterating through image annotations
            for img_ann in img_anns:
                # Normalize keypoints
                try:
                    normalized_kps, normalized_box = self.normalize_keypoints(img_ann["keypoints"], img_ann["bbox"])

                    # Checking feets visibility (for scientific purpose only)
                    # Feets are last 2 sets of kps
                    if normalized_kps[-1] != 0 and normalized_kps[-4] != 0:
                        # Computing % of visible kps
                        if self.__kps_visibility_check(normalized_kps[:-6], threshold):
                            logging.debug(f"[ImageParse]: ACCEPTED Img {img_id}")

                            # Computing distance between 2 feets
                            target = [(normalized_kps[-3] + normalized_kps[-6]) / 2,
                                      (normalized_kps[-5] + normalized_kps[-2]) / 2]

                            # Adding accepted data to parsed_data
                            parsed_data.append(
                                [origin, img_id, img_ann["id"], normalized_box, normalized_kps[:-6], target])
                        else:
                            logging.debug(
                                f"[ImageParse]: REJECTED, Not enough kps visible for image {img_id}")
                    else:
                        logging.debug(
                            f"[ImageParse]: REJECTED, Feets not visible for image {img_id}")

                except Exception as e:
                    print(f"Error while processing image {img_id} | pedestrian {img_ann['id']}: {e}")
                    pass

        return parsed_data

    # -----------------------------------------------------------------------------
    # parse_annotation_file
    # -----------------------------------------------------------------------------
    def parse_annotation_file(self, ann_file_path, cat_names=None, threshold=70, origin="coco") -> list:
        """ Parse the given annotation file.

        Args:
            ann_file_path (str): path to the Coco db annotation file default is empty
            cat_names (list of str): Image Categories (i.e, person), default None
            threshold (int): Required % of visible keypoints, default 70

        Returns:
            list of accepted data:
                each data points are in the form [img_id, annotation_id, keypoints, target]
                where keypoints are a list of keypoints without the feets keypoints
                where target is the distance between 2 feets
        Raises:
            FileExistsError: if file path is erroneous

        Note:
            Calling this method, resets all previously parsed data. Internal parsed data
            will be set to the output of this method
        """
        # Sanity check
        if not os.path.isfile(ann_file_path):
            raise FileExistsError(
                f"File {ann_file_path} does not exist, exiting..")

        # Initializing Coco DB
        coco_db = COCO(ann_file_path)

        # Cat Name check
        if cat_names is None:
            raise Exception(f"Unsupported category names: {cat_names}")

        cat_ids = coco_db.getCatIds(catNms=cat_names)
        img_ids = coco_db.getImgIds(catIds=cat_ids)

        # Parsing selected images
        parsed_data = self.__parse_images(coco_db, img_ids, threshold, origin)

        return parsed_data

    @staticmethod
    def apply_keypoints_occlusion(keypoints, weight_position="", weight_value=0.7,
                                  min_visible_threshold=5, noise_level=[]):
        """
        Applies occlusion to keypoints and normalizes both keypoints and target based on specified parameters.

        Args:
            body_ranges:
            noise_level:
            keypoints (list): List of keypoints for an image, where each keypoint has 3 values (x, y, visibility).
            weight_position (str): "lower_body", "upper_body", or "" for random occlusion.
            weight_value (float): Weight value to determine occlusion probability.
            min_visible_threshold (int): Minimum number of visible keypoints in an image.

        Returns:
            tuple: Tuple containing occluded and normalized keypoints.
        """
        visible_count = sum(1 for i in range(2, len(keypoints), 3) if keypoints[i] > 0)
        if not noise_level:
            noise_level = [0.01] * (len(keypoints) // 3)

        # If already below the threshold, return original keypoints and target
        if visible_count <= min_visible_threshold:
            return keypoints

        # Define ranges for upper and lower body keypoints
        upper_body_range = range(0, 11)
        lower_body_range = range(11, len(keypoints))

        occluded_keypoints = keypoints.copy()
        for i in range(0, len(keypoints), 3):
            if visible_count <= min_visible_threshold:
                break  # Stop if we reach the visibility threshold

            visibility = keypoints[i+2]
            should_occlude = False

            if visibility > 0:
                # Determine if the keypoint should be occluded based on weight_position and weight_value
                if weight_position == "lower_body" and (i // 3) in lower_body_range or \
                        weight_position == "upper_body" and (i // 3) in upper_body_range:
                    should_occlude = random.random() < weight_value
                else:
                    should_occlude = random.random() < (1 - weight_value)

            if should_occlude:
                visible_count -= 1
                # Generate noise for x and y
                noise_x = random.uniform(-noise_level[i // 3], noise_level[i // 3])
                noise_y = random.uniform(-noise_level[i // 3], noise_level[i // 3])

                # Apply noise and set visibility to 0
                occluded_keypoints[i] = keypoints[i] + noise_x if keypoints[i] != 0 else 0
                occluded_keypoints[i + 1] = keypoints[i + 1] + noise_y if keypoints[i] != 0 else 0
                occluded_keypoints[i + 2] = 0

        return occluded_keypoints

    @staticmethod
    def apply_box_occlusion(keypoints, bbox, target, occlusion_chance=0.8, box_scale_factor=(0.5, 1)):
        """
        Applies occlusion to keypoints and the target by modifying the bounding box.

        Args:
            keypoints (list): List of keypoints for an image.
            bbox (list): Bounding box for an image [x, y, width, height].
            target (list): Target keypoint to be normalized alongside the keypoints.
            occlusion_chance (float): Probability of applying occlusion to the bounding box.
            box_scale_factor (tuple): Range (min, max) for the scaling factor of occlusion.

        Returns:
            tuple: Tuple containing occluded and normalized keypoints, occluded bounding box, and normalized target.
        """
        if isinstance(keypoints, str):
            keypoints = eval(keypoints)
        if isinstance(bbox, str):
            bbox = eval(bbox)
        if isinstance(target, str):
            target = eval(target)

        occluded_box = bbox.copy()  # Make a copy of the bbox to avoid modifying the original
        if random.random() < occlusion_chance:
            # Randomly scale the height of the box for occlusion
            scale_factor = random.uniform(*box_scale_factor)
            occluded_box[3] *= scale_factor  # Apply occlusion by modifying the height

        # Normalize keypoints and target with the occluded box
        normalized_keypoints, _ = ImageProcessor.normalize_keypoints(keypoints,
                                                                     [0, 0, occluded_box[2], occluded_box[3]])

        # TODO: Adding visibility data to the target to be able to normalize it
        #       and then removing it (this is a temp solution)
        new_target = target + [2]
        normalized_target, _ = ImageProcessor.normalize_keypoints(new_target,
                                                                  [0, 0, occluded_box[2], occluded_box[3]])

        return occluded_box, normalized_keypoints, normalized_target[:-1]

    # -----------------------------------------------------------------------------
    # apply_keypoints_occlusion
    # -----------------------------------------------------------------------------
    @staticmethod
    def apply_keypoints_occlusion_tensor(inputs, weight_position="", weight_value=0.7, min_visible_threshold=5, noise_level=[]):
        """Applies occlusion to keypoints tensors based on visibility and specified biasing towards body parts.

        Args:
            inputs (Tensor): A tensor of shape (N, M, 3) where N is the batch size, M is the number of keypoints
                             per sample, and each keypoint is represented by 3 values (x, y, visibility).
            weight_position (str): A string to bias the occlusion towards "lower_body" or "upper_body". An empty
                                   string indicates no bias, leading to completely random occlusion.
            weight_value (float): The probability of occluding keypoints. For biased occlusion, this applies to
                                  the specified body part, with the complementary probability applying to the rest.
            min_visible_threshold (int): The minimum number of keypoints that must remain visible after occlusion.
                                         If a sample already has non-visible keypoints below this threshold, it is
                                         skipped to avoid excessive occlusion.

        Returns:
            Tensor: A tensor of the same shape as 'inputs' with occluded keypoints based on the specified parameters.
        """
        occluded_inputs = inputs.clone()
        if not noise_level:
            noise_level = [0.01] * (inputs.size(1))

        for idx, keypoints in enumerate(occluded_inputs):
            keypoints_reshaped = keypoints.view(-1, 3)  # Reshape to have 3 elements per row
            non_visible_count = torch.sum(keypoints_reshaped[:, 2] == 0).item()

            # Skip this keypoints if non-visible keypoints exceed the threshold
            if non_visible_count > min_visible_threshold:
                continue

            for i in range(keypoints_reshaped.size(0)):
                if non_visible_count > min_visible_threshold:
                    break  # Stop if we reach the visibility threshold

                if keypoints_reshaped[i, 2] > 0:  # Check if the keypoint is visible
                    if (weight_position == "lower_body" and i in range(11, len(keypoints_reshaped))) or \
                            (weight_position == "upper_body" and i in range(0, 11)):
                        occlusion_chance = weight_value
                    else:
                        occlusion_chance = 1 - weight_value

                    if random.random() < occlusion_chance:
                        noise = torch.randn(2) * noise_level[i]  # Generate random noise for x and y
                        x_noisy = keypoints_reshaped[i, 0] + noise[0]
                        y_noisy = keypoints_reshaped[i, 1] + noise[1]

                        # Apply noise and set visibility to 0
                        occluded_inputs[idx, 3 * i:3 * i + 3] = torch.tensor([x_noisy, y_noisy, 0], dtype=torch.float32)
                        non_visible_count += 1

        return occluded_inputs

    @staticmethod
    def apply_box_occlusion_tensor(inputs, boxes, targets, occlusion_chance=0.8, box_scale_factor=(0.5, 1)):
        """Applies box occlusion to a batch of keypoints and targets by scaling the bounding boxes.

         Args:
             inputs (Tensor): A tensor of keypoints for each data point in the batch, with shape (N, M, 3),
                              where N is the batch size, M is the number of keypoints, and each keypoint is
                              represented by 3 values (x, y, visibility).
             boxes (Tensor): A tensor of bounding boxes for each data point in the batch, with shape (N, 4),
                             where each box is represented by 4 values (x_min, y_min, width, height).
             targets (Tensor): A tensor of target keypoints for each data point in the batch, with shape (N, 2),
                               where each target is represented by 2 values (x, y).
             occlusion_chance (float): The probability of applying box occlusion to a given sample.
             box_scale_factor (tuple): A range (min, max) for the scaling factor to be applied to the height of
                                       the bounding box, simulating occlusion.

         Returns:
             tuple: A tuple containing three tensors:
                    - The tensor of occluded bounding boxes (N, 4).
                    - The tensor of occluded and re-normalized keypoints (N, M, 3).
                    - The tensor of re-normalized target keypoints (N, 2).
         """
        # Get currently used device
        device = inputs.device

        occluded_inputs = []
        occluded_targets = []
        occluded_boxes = []

        for keypoints, box, target in zip(inputs, boxes, targets):
            # Cloning box to avoid mutation
            box_occluded = box.clone()

            if random.random() < occlusion_chance:
                # Randomly scale the height of the box for occlusion
                scale_factor = random.uniform(*box_scale_factor)
                box_occluded[3] *= scale_factor

            # Normalize keypoints with the occluded box
            # Convert keypoints to list, normalize, and convert back to tensor
            normalized_kps_list, _ = ImageProcessor.normalize_keypoints(keypoints.cpu().tolist(),
                                                                        [0,
                                                                         0,
                                                                         box_occluded[2].item(),
                                                                         box_occluded[3].item()])
            normalized_kps = torch.tensor(normalized_kps_list, dtype=torch.float32, device=device)

            # Temporarily add a visibility value to target for normalization
            target_with_visibility = torch.cat((target, torch.tensor([2.0], dtype=torch.float32, device=device)))

            # Normalize target with the occluded box
            normalized_target_list, _ = ImageProcessor.normalize_keypoints(target_with_visibility.cpu().tolist(),
                                                                           [0,
                                                                            0,
                                                                            box_occluded[2].item(),
                                                                            box_occluded[3].item()])

            # Convert back to tensor and remove the temporary visibility value
            normalized_target = torch.tensor(normalized_target_list[:-1], dtype=torch.float32).to(device)

            occluded_inputs.append(normalized_kps)
            occluded_targets.append(normalized_target)
            occluded_boxes.append(box_occluded)

        # Convert lists to tensors for consistency with PyTorch operations
        occluded_inputs_tensor = torch.stack(occluded_inputs).to(device)
        occluded_targets_tensor = torch.stack(occluded_targets).to(device)
        occluded_boxes_tensor = torch.stack(occluded_boxes).to(device)

        return occluded_boxes_tensor, occluded_inputs_tensor, occluded_targets_tensor