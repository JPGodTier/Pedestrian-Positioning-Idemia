from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from mmdeploy_runtime import PoseDetector


def load_fasterrcnn(device):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    model.eval()
    return model


def load_pose_detector(model_path, device_name="cpu", device_id=0):
    return PoseDetector(model_path=model_path, device_name=device_name, device_id=device_id)
