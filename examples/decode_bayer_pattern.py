import os
import cv2
import numpy as np
from pathlib import Path
import torch
from torchvision.utils import save_image
from scipy.interpolate import griddata

if __name__ == "__main__":
    # scene = "truck"  # rggb
    scene = "garage"  # grbg

    root_path = Path(
        f"/run/determined/workdir/home/gsplat/examples/data/sci_nerf/vggsfm/real/{scene}_colored"
    )
    masks_path = root_path / "mask.npy"
    meas_path = root_path / "meas.npy"

    # read in SCI w/ bayer pattern
    masks = np.load(masks_path)  # (N, H, W) ~ (0, 1)
    meas = np.load(meas_path)  # (H, W)

    # Normalize the image to the range (0, 65535) if necessary
    min_val = np.min(meas)
    max_val = np.max(meas)

    # Avoid division by zero if the image is already within the range (0, 65535)
    if max_val != min_val:
        normalized_bayer_image = (
            (meas - min_val) / (max_val - min_val) * 65535
        ).astype(np.uint16)
    else:
        normalized_bayer_image = (meas * 65535).astype(np.uint16)

    # Perform demosaicing using OpenCV
    rgb_image = cv2.cvtColor(
        normalized_bayer_image, cv2.COLOR_BAYER_GR2BGR
    )  # for garage
    # rgb_image = cv2.cvtColor(normalized_bayer_image, cv2.COLOR_BAYER_RG2BGR)  # truck

    # Reverse the normalization to get the original scale of values
    rgb_image = rgb_image.astype(np.float32) / 65535 * (max_val - min_val) + min_val

    rgb_image = torch.Tensor(rgb_image)
    masks = torch.Tensor(masks)
    H, W, _ = rgb_image.shape

    masks_sum = masks.sum(0)
    rgb_image = rgb_image / masks_sum.unsqueeze(-1)

    decoded_imgs = []

    for mask in masks:
        decoded_img = rgb_image * mask.unsqueeze(-1)

        # x, y = torch.where(mask == 1)
        x, y = torch.where(mask >= 0.6)
        points = np.stack((x, y)).T

        decoded_img_values = decoded_img[points[:, 0], points[:, 1], :]

        grid_x, grid_y = np.mgrid[0:H, 0:W]

        # grid_z0 = griddata(
        #     points, decoded_img_values, (grid_x, grid_y), method="nearest"
        # )  # use nearest since it will not produce black pixels at corners

        grid_z0 = griddata(
            points, decoded_img_values, (grid_x, grid_y), method="linear"
        )  # use nearest since it will not produce black pixels at corners

        decoded_img_tensor = torch.tensor(grid_z0).permute(2, 0, 1)

        decoded_imgs.append(decoded_img_tensor)

    decoded_imgs = torch.stack(decoded_imgs)

    # folder_path = f"./norm_decoded_imgs_{scene}_linear"
    folder_path = root_path / "images"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created at {folder_path}")
    else:
        print(f"Folder already exists at {folder_path}")

    # save images ina folder
    for idx, decoded_img in enumerate(decoded_imgs):
        save_image(decoded_img, os.path.join(folder_path, f"{idx}.png"))
