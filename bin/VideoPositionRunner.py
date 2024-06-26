import cv2
import torch
import os
from src.Pipelines.Pipeline3dProcessor import Pipeline3DProcessor
from src.Common.utils import save_depth_graph, save_depth_graph2

ground_truths = {
    0: 3.0,
    5: 3.0,
    40: 2.4,
    70: 1.8,
    95: 1.2,
    172: 1.5,
    200: 2.1,
    220: 2.7,
    245: 3.3,
}


# -----------------------------------------------------------------------------
# Visualisation methods
# -----------------------------------------------------------------------------
def resize_frame(frame, width_scale=100, height_scale=100):
    width = int(frame.shape[1] * width_scale / 100)
    height = int(frame.shape[0] * height_scale / 100)
    dim = (width, height)
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return resized


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model_path = "data/CNN_Model/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.zip"
    mlp_model_path = os.path.join("models", "20240616_205325_LR0.0001_BS16", "final_model_epoch_44_rmse_0.0671.pth")

    calibration_path = os.path.join(os.getcwd(), "config", "calibration_chessboard.yaml")
    mlp_settings = {"input_size": 45, "output_size": 2, "layers": [256, 128, 64, 32]}
    pipeline_processor = Pipeline3DProcessor(device, cnn_model_path, mlp_model_path, calibration_path, mlp_settings)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Video file could not be opened.")
        return

    # Setup Depth logging variables
    depth_predictions = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame could not be read.")
            break

        predictions, scores, keypoints_list = pipeline_processor.detect_people(frame)

        if keypoints_list:
            keypoints, bbox = keypoints_list
            x1, y1, width, height = bbox

            cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height), (0, 255, 0), thickness=3)

            if keypoints:
                feet_2d_position = pipeline_processor.infer_2d_position(keypoints[:-6])
                abs_x, abs_y = int(x1 + feet_2d_position[0] * width), int(y1 + feet_2d_position[1] * height)
                cv2.circle(frame, (abs_x, abs_y), 10, (0, 0, 255), -1)

                world_point = pipeline_processor.calibration.estimate_3d_point_pinv(abs_x, abs_y)
                world_coords = f"Feet 3D: ({world_point[0]:.2f}, {world_point[1]:.2f}, {world_point[2]:.2f}, {frame_count})"
                cv2.putText(frame, world_coords, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 255), 5)

                # Logging Depth Estimation
                depth_predictions.append(world_point[2])

                for i in range(0, len(keypoints), 3):
                    norm_x, norm_y, conf = keypoints[i], keypoints[i + 1], keypoints[i + 2]
                    abs_x, abs_y = int(x1 + norm_x * width), int(y1 + norm_y * height)
                    cv2.circle(frame, (abs_x, abs_y), 10, (0, 255, 0), -1)

        frame = resize_frame(frame, width_scale=55, height_scale=30)
        cv2.imshow('Frame', frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Generate Depth plot
    save_depth_graph2(depth_predictions, ground_truths, model_path="DepthStudy", name="DepthAnalysis")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("video-testing\\out_1_blurred.mp4")
