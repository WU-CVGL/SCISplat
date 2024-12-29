import os
import numpy as np
from pathlib import Path

import datasets.colmap_parsing_utils as colmap_utils
from trajectory_evaluation import (
    plot_trajectories2D,
    align_umeyama,
    compute_absolute_error_translation,
    fig_to_array,
)


def read_tum_poses(file_path):
    poses = []
    with open(file_path, "r") as file:
        for line in file:
            values = line.strip().split()
            if len(values) == 8:
                _, tx, ty, tz, qx, qy, qz, qw = map(float, values)
                # Create transformation matrix
                t_matrix = quaternion_to_transformation_matrix(
                    tx, ty, tz, qx, qy, qz, qw
                )
                poses.append(t_matrix)
    poses = np.stack(poses, axis=0)
    return poses


def quaternion_to_transformation_matrix(tx, ty, tz, qx, qy, qz, qw):
    # Normalize quaternion
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    # Create rotation matrix
    rotation_matrix = np.array(
        [
            [
                1 - 2 * qy * qy - 2 * qz * qz,
                2 * qx * qy - 2 * qz * qw,
                2 * qx * qz + 2 * qy * qw,
            ],
            [
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx * qx - 2 * qz * qz,
                2 * qy * qz - 2 * qx * qw,
            ],
            [
                2 * qx * qz - 2 * qy * qw,
                2 * qy * qz + 2 * qx * qw,
                1 - 2 * qx * qx - 2 * qy * qy,
            ],
        ]
    )

    # Create transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = [tx, ty, tz]

    return transformation_matrix


if __name__ == "__main__":
    root_path = Path("/run/determined/workdir/home/gsplat/examples")
    scene = "factory"

    gt_poses_path = root_path / Path(f"scinerf_poses/{scene}/groundtruth.txt")
    poses_path = root_path / Path(f"scinerf_poses/{scene}/ours.txt")

    # read in poses to be aligned (TUM format)
    camtoworlds_gt = read_tum_poses(gt_poses_path)
    camtoworlds = read_tum_poses(poses_path)

    # align to GT
    traj_gt = camtoworlds_gt[:, :3, -1]
    traj = camtoworlds[:, :3, -1]

    # log their 3D trajectories to writer as an image
    s, R, t = align_umeyama(traj_gt, traj)
    traj_aligned = (s * (R @ traj.T)).T + t

    # plot them post alignment
    fig = plot_trajectories2D(traj_gt, traj_aligned)
    img_pil, img_array = fig_to_array(fig)

    # compute ATEs
    ate = compute_absolute_error_translation(traj_gt, traj_aligned)
    print(f"ATE: {ate}")
