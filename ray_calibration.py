from dataset_generator import generate_dateset
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def ray_calibrate(decode_maps, decode_masks, target_size):
    # Get input info.
    decode_maps, decode_masks, target_size = torch.tensor(decode_maps, dtype=torch.float32), torch.tensor(decode_masks,  dtype=torch.uint8), torch.tensor(target_size, dtype=torch.float32)
    image_size = decode_maps.shape[1:3]
    assert (image_size == decode_masks.shape[1:])
    n_targets = len(decode_maps)
    assert (n_targets == len(decode_masks))

    # Define trainable parameters.
    target_pose_parameters = nn.Parameter(torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.float32).repeat(n_targets, 1))
    target_pose_parameters[0][0].clamp(0, 0)
    target_pose_parameters[0][1].clamp(0, 0)
    target_pose_parameters[0][2].clamp(0, 0)
    target_pose_parameters[0][3].clamp(0, 0)
    target_pose_parameters[0][4].clamp(0, 0)
    target_pose_parameters[0][5].clamp(1, 1)
    ray_parameters = nn.Parameter(torch.zeros(*image_size, 4)) # [480, 640, 4]. x, y, u, v

    # Optimizer.
    optimizer = optim.Adam([target_pose_parameters, ray_parameters], lr=5e-2)

    # Optimization loop.
    for epoch in range(10000):
        # Get target pose transforms.
        transforms_camera_to_target = torch.cat([
            euler_to_rotation_matrix(target_pose_parameters),
            target_pose_parameters[:, 3:].unsqueeze(2)], dim=2)

        # Compute loss
        loss = torch.tensor(0.0)
        for i_target in range(n_targets):

            decode_map = decode_maps[i_target] # 480 * 640 * 2
            decode_mask = decode_masks[i_target] # 480 * 640
            transform_camera_to_target = transforms_camera_to_target[i_target]

            # Compute target point local positions.
            target_points_local = torch.cat([
                decode_map * target_size.reshape(1, 1, 2),
                torch.zeros(*decode_mask.shape, 1, dtype=decode_map.dtype, device=decode_map.device),
                torch.ones(*decode_mask.shape, 1, dtype=decode_map.dtype, device=decode_map.device)], dim=-1)

            # Compute target point global positions.
            target_points_global = target_points_local @ transform_camera_to_target.T

            # Compute loss.
            x_error = (target_points_global[..., 0] - (ray_parameters[..., 2] * target_points_global[..., 2]) - ray_parameters[..., 0]).squeeze() * decode_mask
            y_error = (target_points_global[..., 1] - (ray_parameters[..., 3] * target_points_global[..., 2]) - ray_parameters[..., 1]).squeeze() * decode_mask
            loss += ((x_error ** 2).sum() + (y_error ** 2).sum())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(epoch, loss.item())

        # Visualize.
        if epoch % 100:
            continue

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ray_parameters_flat = ray_parameters.reshape(-1, 4).detach().numpy()[::1000, :]
        #print(ray_parameters_flat.shape)
        ax.set_xlim([-5.0, 5.0])
        ax.set_ylim([-5.0, 5.0])
        ax.set_zlim([-5.0, 5.0])

        # Draw rays.
        ax.quiver(
            ray_parameters_flat[:, 0],
            ray_parameters_flat[:, 1],
            np.zeros(len(ray_parameters_flat)),
            ray_parameters_flat[:, 2],
            ray_parameters_flat[:, 3],
            np.ones(len(ray_parameters_flat)),
            length=3.0,
        )

        # Draw targets.
        target_vertices_local = np.array([
            [0, 0, 0, 1],
            [target_size[0], 0, 0, 1],
            [target_size[0], target_size[1], 0, 1],
            [0, target_size[1], 0, 1],
        ])
        target_vertices_global = np.einsum(
            'p j, b i j -> b p i',
            target_vertices_local,
            transforms_camera_to_target.detach().numpy())
        for target_vertices in target_vertices_global:
            ax.add_collection3d(Poly3DCollection(
                [target_vertices[:, :3]], facecolors='cyan', edgecolors='black', linewidths=1, alpha=0.7))

        plt.show()
        plt.close()



def euler_to_rotation_matrix(euler):
    rx, ry, rz = euler[..., 0], euler[..., 1], euler[..., 2]
    cos = torch.cos
    sin = torch.sin
    cx, cy, cz = cos(rx), cos(ry), cos(rz)
    sx, sy, sz = sin(rx), sin(ry), sin(rz)
    rotation_matrices = torch.stack([
        torch.stack([cy*cz, -cy*sz, sy], dim=-1),
        torch.stack([sx*sy*cz + cx*sz, -sx*sy*sz + cx*cz, -sx*cy], dim=-1),
        torch.stack([-cx*sy*cz + sx*sz, cx*sy*sz + sx*cz, cx*cy], dim=-1)
    ], dim=-2)
    return rotation_matrices