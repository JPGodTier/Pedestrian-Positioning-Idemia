import csv
import pandas as pd
import os
import numpy as np
import math
import yaml
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO
from tabulate import tabulate
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# save_to_csv
# -----------------------------------------------------------------------------
def save_to_csv(file, headers, data_list):
    """ Saving given headers & data to given file.

    Args:
        file: path to the output file
        headers: CSV file headers
        data_list: Data to be saved

    Returns:
        None
    """
    with open(file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Writing headers & data
        writer.writerow(headers)
        for row in data_list:
            writer.writerow(row if isinstance(row, list) else [row])


# -----------------------------------------------------------------------------
# log_model_results
# -----------------------------------------------------------------------------
def log_model_results(performance_data, csv_file="./results/model_performance.csv"):
    """
    Records the performance data into a CSV file, using the experiment timestamp for consistency.

    Args:
        performance_data (dict): Performance data to log.
        csv_file (str): Path to the CSV file for logging performance data.
    """
    results_dir = os.path.dirname(csv_file)
    os.makedirs(results_dir, exist_ok=True)

    # Dataframe Init
    df_new_entry = pd.DataFrame([performance_data])

    # Write performance data to csv
    try:
        if os.path.isfile(csv_file):
            df_existing = pd.read_csv(csv_file, sep=";")
            df_final = pd.concat([df_existing, df_new_entry], ignore_index=True, sort=False)
        else:
            df_final = df_new_entry

        df_final.to_csv(csv_file, index=False, sep=";")
    except Exception as e:
        print(f"Error recording results: {e}")


# -----------------------------------------------------------------------------
# visualize_csv_stats
# -----------------------------------------------------------------------------
def visualize_csv_stats(file_path):
    """ Visualizes statistics of a keypoints dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        stats: dictionary containing the csv detailed statistics
    """
    keypoint_names = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                      "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
                      "left_knee", "right_knee"]

    # Load the CSV file
    df = pd.read_csv(file_path)

    # Convert keypoints from string to list
    df['keypoints'] = df['keypoints'].apply(eval)

    # Calculate visibility for each keypoint
    df['visible_keypoints'] = df['keypoints'].apply(lambda kps: [kps[i*3+2] for i in range(len(keypoint_names))])
    df['occluded_keypoints'] = df['visible_keypoints'].apply(lambda vis: [1 if v == 0 else 0 for v in vis])

    # Calculate the most frequently occluded keypoint
    occlusion_counts = [sum(kps[i] for kps in df['occluded_keypoints']) for i in range(len(keypoint_names))]
    most_occluded_index = occlusion_counts.index(max(occlusion_counts))
    most_frequent_occluded_kp = keypoint_names[most_occluded_index] if occlusion_counts[most_occluded_index] > 0 else None

    stats = {
        'Total images': len(df),
        'Average keypoints occluded per image': math.ceil(sum(occlusion_counts) / len(df)),
        'Max visible keypoints': df['visible_keypoints'].apply(lambda kps: sum(v > 0 for v in kps)).max(),
        'Min visible keypoints': df['visible_keypoints'].apply(lambda kps: sum(v > 0 for v in kps)).min(),
        'Percentage of images with occlusion': round(100 * sum(any(v == 0 for v in vis) for vis in df['visible_keypoints']) / len(df), 2),
        'Most frequently occluded keypoint': most_frequent_occluded_kp,
    }

    # Print table & return results
    print(tabulate(stats.items(), headers=["Statistic", "Value"], floatfmt=".2f", stralign="left"))
    return stats


# -----------------------------------------------------------------------------
# load_config
# -----------------------------------------------------------------------------
def load_config(config_path):
    with open(config_path, 'r') as file:
        p_config = yaml.safe_load(file)
    return p_config


# -----------------------------------------------------------------------------
# visualize_specific_pedestrian_local
# -----------------------------------------------------------------------------
def visualize_specific_pedestrian_local(annotation_file, images_folder_path, image_id, pedestrian_id, pred, truth):
    # Initialize COCO api
    coco = COCO(annotation_file)

    # Load image metadata
    img = coco.loadImgs(int(image_id))[0]

    # Construct image path
    # Assuming the image filename is formatted with 12 digits
    img_filename = f"{str(image_id).zfill(12)}.jpg"
    img_path = os.path.join(images_folder_path, img_filename)

    # Load the image from the local path
    image = Image.open(img_path)

    # Display the image
    plt.imshow(image)
    plt.axis('off')

    ax = plt.gca()
    anns = coco.loadAnns(int(pedestrian_id))

    for ann in anns:
        # Draw keypoints
        if "keypoints" in ann:
            keypoints = ann["keypoints"]
            for i in range(0, len(keypoints), 3):
                kp_x, kp_y, v = keypoints[i], keypoints[i + 1], keypoints[i + 2]
                # Draw only if keypoint is labeled and visible
                if v == 2:
                    ax.plot(kp_x, kp_y, 'o', color='blue', markersize=3)

        x0, y0, width, height = ann["bbox"]
        denormalized_pred_x = (pred[0] * width) + x0
        denormalized_pred_y = (pred[1] * height) + y0
        denormalized_truth_x = (truth[0] * width) + x0
        denormalized_truth_y = (truth[1] * height) + y0
        ax.plot(denormalized_pred_x, denormalized_pred_y, 'o', color='red', markersize=6)
        ax.plot(denormalized_truth_x, denormalized_truth_y, '*', color='green', markersize=5)

    plt.title(f"Image ID: {img['id']}, Pedestrian ID: {pedestrian_id}")

    plt.show()


# -----------------------------------------------------------------------------
# calculate_euclidean_distance
# -----------------------------------------------------------------------------
def calculate_euclidean_distance(pred, truth):
    return np.sqrt(np.sum((np.array(pred) - np.array(truth))**2))


# -----------------------------------------------------------------------------
# load_mlp_model
# -----------------------------------------------------------------------------
def load_mlp_model(device, model_path, input_size, output_size, layers, is_eval=True):
    from src.Models.Mlp import MLP
    model = MLP(input_size, output_size, layers)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Load the model in eval mode if required
    if is_eval:
        model.eval()

    model.to(device)
    return model


# -----------------------------------------------------------------------------
# save_depth_graph
# -----------------------------------------------------------------------------
def save_depth_graph(depth_predictions, ground_truths, model_path, name):
    frames = list(range(len(depth_predictions)))

    # Graph Theme
    template_theme = "seaborn"

    # Preparing data
    gt_x = [frame for frame in frames if frame in ground_truths]
    gt_y = [ground_truths[frame] for frame in gt_x]
    pred_y = [depth_predictions[i] for i in gt_x]

    # Calculate percentage errors
    percentage_errors = [abs(pred - gt) / gt * 100 for pred, gt in zip(pred_y, gt_y)]
    max_error = max(percentage_errors) if percentage_errors else 0
    avg_error = sum(percentage_errors) / len(percentage_errors) if percentage_errors else 0

    # Creating hover text for each ground truth point
    hover_texts = [f"Error: {error:.2f}%" for error in percentage_errors]

    # Trace for Depth Predictions
    trace_depth_pred = go.Scatter(
        x=frames, y=depth_predictions,
        mode='lines+markers',
        name='Predicted Depth',
        marker=dict(color='LightSkyBlue'),
        line=dict(color='DeepSkyBlue')
    )

    # Trace for Ground Truth Depth with hover info
    trace_depth_gt = go.Scatter(
        x=gt_x,
        y=gt_y,
        mode='markers',
        name='Ground Truth Depth',
        marker=dict(color='Gold', size=12, line=dict(color='Black', width=2)),
        text=hover_texts,
        hoverinfo='text+y'
    )

    # Layout settings with annotations for max and average errors
    annotations = [
        dict(xref='paper', yref='paper', x=0, y=0.05,
             text='Max Error: {:.2f}%'.format(max_error),
             showarrow=False, font=dict(color="Red", size=14)),
        dict(xref='paper', yref='paper', x=0, y=0,
             text='Average Error: {:.2f}%'.format(avg_error),
             showarrow=False, font=dict(color="Green", size=14))
    ]

    layout = go.Layout(
        title=f'{name} Depth Estimation Performance',
        xaxis=dict(title='Frame Number'),
        yaxis=dict(title='Depth (meters)', autorange=True),
        hovermode='closest',
        template=template_theme,
        legend=dict(title='Metrics', x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.5)'),
        margin=dict(l=40, r=40, t=40, b=40),
        annotations=annotations
    )
    fig = go.Figure(data=[trace_depth_pred, trace_depth_gt], layout=layout)

    # Save to HTML file
    fig.write_html(os.path.join(model_path, f"{name}_depth_performance.html"))
