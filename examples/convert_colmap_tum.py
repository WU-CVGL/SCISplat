import numpy as np

from pathlib import Path
import datasets.colmap_parsing_utils as colmap_utils
from trajectory_evaluation import transformation_to_tum_format, save_to_tum_file

if __name__ == "__main__":
    # read in colmap poses in transformation matrix format
    root_path = Path("/run/determined/workdir/home/gsplat/examples")
    scene = "cozy2room"

    colmap_dir = (
        root_path
        / Path(
            "data/sci_nerf/ablation_study/initial_points/cozy2room_nearest_qf8_shared_pts4096"
        )
        / "sparse/0/"
    )

    # read in gt camera poses
    if (colmap_dir / "images_gt.bin").exists():
        im_id_to_image_gt = colmap_utils.read_images_binary(
            colmap_dir / "images_gt.bin"
        )
    elif (colmap_dir / "images_gt.txt").exists():
        im_id_to_image_gt = colmap_utils.read_images_text(colmap_dir / "images_gt.txt")
    else:
        im_id_to_image_gt = None

    ordered_im_id = sorted(im_id_to_image_gt.keys())

    w2c_mats_gt = []
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    for im_id in ordered_im_id:
        im_gt = im_id_to_image_gt[im_id]
        rot_gt = colmap_utils.qvec2rotmat(im_gt.qvec)
        trans_gt = im_gt.tvec.reshape(3, 1)
        w2c_gt = np.concatenate([np.concatenate([rot_gt, trans_gt], 1), bottom], axis=0)
        w2c_mats_gt.append(w2c_gt)

    image_names = [im_id_to_image_gt[k].name for k in ordered_im_id]
    inds = np.argsort(image_names)

    w2c_mats_gt = np.stack(w2c_mats_gt, axis=0)
    camtoworlds_gt = np.linalg.inv(w2c_mats_gt)
    camtoworlds_gt = camtoworlds_gt[inds]  # (N, 4, 4)

    # convert transformation matrix format to tum format
    timestamps = np.array(
        [
            0.000000000000000000e00,
            1.428571492433547974e-01,
            2.857142984867095947e-01,
            4.285714626312255859e-01,
            5.714285373687744141e-01,
            7.142857313156127930e-01,
            8.571428656578063965e-01,
            1.000000000000000000e00,
        ]
    )
    tum_poses = transformation_to_tum_format(camtoworlds_gt, timestamps)
    save_to_tum_file(tum_poses, colmap_dir / "poses_gt.txt")
