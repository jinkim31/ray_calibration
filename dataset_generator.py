import cv2
import numpy as np

def generate_dateset(camera_matrix, rotations_camera_to_target, translations_camera_to_target, target_size, image_size):
    """
    :param camera_matrix: Numpy array of size [3, 3].
    :param rotations_camera_to_target: Rotation matrices from camera and calibration target poses. Numpy array of size [n_calibration_targets, 3, 3].
    :param translations_camera_to_target: Translation vectors from camera and calibration target poses. Numpy array of size [n_calibration_targets, 3].
    :param target_size: Calibration target size in meters, Array-like of size [2].
    :param image_size: Output image size in pixels. Array-like of size [2].
    :return: Decoded images for each calibration target poses. [n_calibration_targets, image_size[0], image_size[1]].
    """

    # Get transforms from 3D world point to 2D image for each target pose.
    transforms_target_to_image = camera_matrix @ np.concatenate([rotations_camera_to_target, np.expand_dims(translations_camera_to_target, 2)], axis=2)

    # Only use x, y part since z is always 0.
    transforms_target_to_image = transforms_target_to_image[:, :, [0, 1, 3]] # (2, 3, 3)

    # get inverse transforms for inverse warping
    transforms_image_to_target = np.array([np.linalg.inv(t) for t in transforms_target_to_image]) # (n_target_poses, 3, 3)

    # inverse warp the target
    grid_x, grid_y = np.meshgrid(np.arange(image_size[0]),np.arange(image_size[1]), indexing='xy')
    image_pixel_positions = np.stack([
        grid_x, grid_y,
        np.ones(image_size[::-1])
    ], axis=2) # [h, w, 3]
    target_point_positions = np.einsum('nij,hwj->nihw', transforms_image_to_target, image_pixel_positions) # [480, 640, 3]
    target_point_positions = np.transpose(target_point_positions, (0, 2, 3, 1))
    target_point_positions = target_point_positions[..., :2] / target_point_positions[..., 2:3]

    # Make decoded images.
    x = target_point_positions[..., 0]
    y = target_point_positions[..., 1]
    x_mask = (x > 0) & (x < target_size[0])
    y_mask = (y > 0) & (y < target_size[1])
    combined_mask = x_mask & y_mask  # shape: (3, 640, 480)
    decode_maps = np.full_like(target_point_positions, 0.0)  # shape: (3, 640, 480, 2)
    decode_maps[..., 0][combined_mask] = (x[combined_mask] - 0) / target_size[0]
    decode_maps[..., 1][combined_mask] = (y[combined_mask] - 0) / target_size[1]
    decode_masks = np.where(combined_mask, 1.0, 0.0)  # shape: (3, 640, 480)
    return decode_maps, decode_masks
