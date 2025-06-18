from dataset_generator import generate_dateset
from ray_calibration import ray_calibrate, plot
import numpy as np
import cv2
import torch
import time


# System.
focal_length = 320
image_size = [640, 480]
target_size = [1.0, 1.0]
camera_matrix = np.array([
    [focal_length, 0, image_size[0]/2],
    [0, focal_length, image_size[1]/2],
    [0, 0, 1]
], dtype=float)

# Generate dataset with varying calibration target poses.
n_targets = 50
np.random.seed(0) # Set seed for reproducibility.
rotations_camera_to_target, translations_camera_to_target = [], []
for _ in range(n_targets):
    rotations_camera_to_target.append(np.eye(3))
    translations_camera_to_target.append([
        np.random.uniform(-0.5, 0.5) - target_size[0]/2,
        np.random.uniform(-0.5, 0.5) - target_size[1]/2,
        np.random.uniform(2, 3)
    ])
decode_maps, decode_masks = generate_dateset(camera_matrix, rotations_camera_to_target, translations_camera_to_target, target_size, image_size)

# Calibrate.
start = time.time()
ray_parameters, transforms_camera_to_target = ray_calibrate(
    decode_maps, decode_masks, target_size,
    learning_rate=5e-2, stop_loss=1e-1, show_plot=False, verbose=True, device='cuda')

# Plot.
plot(ray_parameters, transforms_camera_to_target, target_size, ray_decimation_rate=100)
print(f"Calibrated in {time.time() - start:.4f} seconds") #CPU: 266.1496 s, GPU: 52.3428 s
