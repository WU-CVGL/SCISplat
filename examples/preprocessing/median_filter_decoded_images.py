"""
This script applys median filter on decoded images from a snapshot compressive image
to remove spiky pixel values.
"""

import os
import sys
import torch
import numpy as np
from torchvision.utils import save_image
from scipy.interpolate import griddata
from img_utils import masked_median_filter, interpolate_img_with_mask

if __name__ == "__main__":
    # read in a decoded image and its mask
    root_path = sys.argv[1]
    scene = sys.argv[2]

    img_path = f"{root_path}/{scene}/meas.npy"
    mask_path = f"{root_path}/{scene}/mask.npy"

    # scene = 'airplants'

    # img_path = f'/run/determined/workdir/home/data/SCINeRF-data/input_data/{scene}/meas.npy'
    # mask_path = f'/run/determined/workdir/home/data/SCINeRF-data/input_data/{scene}/mask.npy'

    # set the kernel size (odd number: 5x5, 7x7, 9x9 ...)
    kernel_size = 5

    comp_img = torch.Tensor(np.load(img_path))

    H, W, _ = comp_img.shape

    masks = torch.Tensor(np.load(mask_path))

    # compressive image normalization
    masks_sum = masks.sum(0)
    comp_img = comp_img / masks_sum.unsqueeze(-1)

    decoded_imgs = []

    for mask in masks:
        decoded_img = comp_img * mask.unsqueeze(-1)

        decoded_imgs.append(decoded_img.permute(2, 0, 1))

    filtered_imgs = []
    # get the filtered decoded image
    for decoded_img, mask in zip(decoded_imgs, masks):
        # NOTE: this function is now slow as fuck, need to boost it up in the future
        filtered_img = masked_median_filter(
            decoded_img, mask, kernel_size=kernel_size
        )  # (3, H, W)

        filtered_img = interpolate_img_with_mask(filtered_img, mask)

        filtered_imgs.append(filtered_img)

    filter_comment = f"median{kernel_size}"
    folder_path = f"{root_path}/{scene}/decoded_imgs_{scene}_nearest_{filter_comment}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created at {folder_path}")
    else:
        print(f"Folder already exists at {folder_path}")

    # save images ina folder
    for idx, filtered_img in enumerate(filtered_imgs):
        save_image(filtered_img, os.path.join(folder_path, f"frame_{idx}.png"))
    pass
