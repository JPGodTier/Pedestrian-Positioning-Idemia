import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
import datetime
import time
import mlflow
import random
import numpy as np

from src.DataLoader.DataLoader import KeypointsDataset, csv_string_to_list
from src.Models.Mlp import MLP
from src.Training.Trainer import train_model
from src.Common.utils import log_model_results, load_config


# Setting the seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 7495
set_seed(seed)


# -----------------------------------------------------------------------------
# prepare_data
# -----------------------------------------------------------------------------
def prepare_data(data_path):
    df = pd.read_csv(data_path)
    df['bbox'] = df['bbox'].apply(csv_string_to_list)
    df['keypoints'] = df['keypoints'].apply(csv_string_to_list)
    df['target'] = df['target'].apply(csv_string_to_list)
    return KeypointsDataset(df)


# -----------------------------------------------------------------------------
# MLFLOW Setup
# -----------------------------------------------------------------------------
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("Visibility Threshold Experiments")

# -----------------------------------------------------------------------------
# MODEL LOGIC
# -----------------------------------------------------------------------------
start_time = time.time()

# Setup
config = load_config(os.path.join(os.getcwd(), "config", "config.yaml"))
device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
print("Running on device: ", device)

models_path = os.path.join(os.getcwd(), "models")
if not os.path.exists(models_path):
    os.makedirs(models_path)

with mlflow.start_run():
    # Creating sub-folder for current run
    exp_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_learning_rate = config['training']['learning_rate']
    exp_batch_size = config['training']['batch_size']
    exp_name = f"{exp_timestamp}_LR{exp_learning_rate}_BS{exp_batch_size}"

    exp_path = os.path.join("models", exp_name)
    os.makedirs(exp_path, exist_ok=True)

    # Train occlusion settings
    occ_chance = config["occultation"]["occlusion_chance"]
    box_scale = config["occultation"]["box_scale_factor"]
    box_scale = (box_scale[0], box_scale[1])
    weight_value = config["occultation"]["weight_value"]
    weight_position = config["occultation"]["weight_position"]
    min_visible_kps = config["occultation"]["min_visible_kps"]
    noise_per_keypoint = config["occultation"]["noise_per_keypoint"]

    occlusion_params = {
        'box_occlusion': {'occlusion_chance': occ_chance, 'box_scale_factor': box_scale},
        'keypoints_occlusion': {'weight_position': weight_position, 'weight_value': weight_value,
                                'min_visible_threshold': min_visible_kps, "noise_level": noise_per_keypoint}}

    # Data Preparation
    train_dataset = prepare_data(config['data']['train_path'])
    val_dataset = prepare_data(config['data']['validation_path'])

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # Model Initialization
    input_size = len(train_dataset[0][2])  # Features
    model = MLP(input_size, config['model']['output_size'], config['model']['layers']).to(device)

    # Loging with MLFLOW
    mlflow.log_param("File Name", exp_name)
    mlflow.log_param("Data Info", "CNN Keypoint Inference")
    mlflow.log_param("Seed", seed)
    mlflow.log_params(config['data'])
    mlflow.log_params(config['occultation'])
    mlflow.log_params(config['training'])

    # Training and retrieving best RMSE
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    loss_function = torch.nn.MSELoss()

    best_rmse, best_epoch = train_model(model,
                                        train_loader,
                                        val_loader,
                                        optimizer,
                                        loss_function,
                                        config['training']['epochs'],
                                        device,
                                        exp_path,
                                        occlusion_params)

    mlflow.log_metric("Best RMSE", best_rmse)
    mlflow.log_metric("Best Epoch", best_epoch)

# -----------------------------------------------------------------------------
# POST-TRAINING ACTIONS
# -----------------------------------------------------------------------------
# Saving Model
print("Saving model & model performances...")
model_filename = f"final_model_epoch_{best_epoch}_rmse_{best_rmse:.4f}.pth"
torch.save(model.state_dict(), os.path.join(exp_path, model_filename))
print(f"Model {model_filename} saved.")

# Saving Model performances
print("Adding Model performances in log file...")
performance_data = {
    "Timestamp": exp_timestamp,
    "Learning_Rate": exp_learning_rate,
    "Batch_Size": exp_batch_size,
    "Model_Layers": config['model']['layers'],
    "Configured Epochs": config['training']['epochs'],
    "Best Epoch": best_epoch,
    "RMSE": best_rmse
}
log_model_results(performance_data, csv_file=os.path.join(models_path, "model_performance_logs.csv"))
print("Model Performance saved.")

end_time = time.time()
print(f"Execution Time: {end_time - start_time:.2f} seconds")