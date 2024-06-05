import cv2
import torch
import os
from src.Pipelines.Pipeline3dProcessor import Pipeline3DProcessor


# -----------------------------------------------------------------------------
# Visualisation methods
# -----------------------------------------------------------------------------
def resize_frame(frame, width_scale=50, height_scale=50):
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
    mlp_model_path = os.path.join("models", "20240516_152621_LR0.0001_BS16", "final_model_epoch_32_rmse_0.0689.pth")
    calibration_path = os.path.join(os.getcwd(), "config", "calibration_chessboard.yaml")
    mlp_settings = {"input_size": 45, "output_size": 2, "layers": [256, 128, 64, 32]}
    pipeline_processor = Pipeline3DProcessor(device, cnn_model_path, mlp_model_path, calibration_path, mlp_settings)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Video file could not be opened.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame could not be read.")
            break

        frame = resize_frame(frame, width_scale=40, height_scale=30)
        predictions, scores, keypoints_list = pipeline_processor.detect_people(frame)

        for box, score, keypoints_info in zip(predictions, scores, keypoints_list):
            keypoints, bbox = keypoints_info
            x1, y1, width, height = bbox
            cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height), (0, 255, 0), 2)

            if keypoints:
                feet_2d_position = pipeline_processor.infer_2d_position(keypoints)
                abs_x, abs_y = int(x1 + feet_2d_position[0] * width), int(y1 + feet_2d_position[1] * height)
                cv2.circle(frame, (abs_x, abs_y), 5, (0, 0, 255), -1)

                world_point = pipeline_processor.calibration.estimate_3d_point_pinv_idemia(abs_x, abs_y)
                world_coords = f"Feet 3D: ({world_point[0]:.2f}, {world_point[1]:.2f}, {world_point[2]:.2f})"
                cv2.putText(frame, world_coords, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                for i in range(0, len(keypoints), 3):
                    norm_x, norm_y, conf = keypoints[i], keypoints[i + 1], keypoints[i + 2]
                    abs_x, abs_y = int(x1 + norm_x * width), int(y1 + norm_y * height)
                    cv2.circle(frame, (abs_x, abs_y), 3, (0, 255, 0), -1)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("video-testing\\out_1_blurred.mov")
