import os
import torch
import numpy as np
from torchvision.utils import save_image
from torchvision.transforms import GaussianBlur
from scipy.interpolate import griddata
import sys

# root_path = '/run/determined/workdir/home/SCI-Gaussians/preprocessing'
# scene = 'kun'

root_path = sys.argv[1]
scene = sys.argv[2]

img_path = f"{root_path}/{scene}/meas.npy"
mask_path = f"{root_path}/{scene}/mask.npy"

comp_img = torch.Tensor(np.load(img_path))
H, W, _ = comp_img.shape

masks = torch.Tensor(np.load(mask_path))

# energy normalization
masks_sum = masks.sum(0)
comp_img = comp_img / masks_sum.unsqueeze(-1)

decoded_imgs = []

for mask in masks:
    decoded_img = comp_img * mask.unsqueeze(-1)

    x, y = torch.where(mask == 1)
    points = np.stack((x, y)).T

    decoded_img_values = decoded_img[points[:, 0], points[:, 1], :]

    grid_x, grid_y = np.mgrid[0:H, 0:W]

    # grid_z0 = griddata(
    #     points, decoded_img_values, (grid_x, grid_y), method="nearest"
    # )  # use nearest since it will not produce black pixels at corners

    grid_z0 = griddata(
        points, decoded_img_values, (grid_x, grid_y), method="linear"
    )  # use nearest since it will not produce black pixels at corners

    decoded_imgs.append(torch.tensor(grid_z0).permute(2, 0, 1))

decoded_imgs = torch.stack(decoded_imgs)

folder_path = f"{root_path}/{scene}/decoded_images"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder created at {folder_path}")
else:
    print(f"Folder already exists at {folder_path}")

# save images ina folder
for idx, decoded_img in enumerate(decoded_imgs):
    save_image(decoded_img, os.path.join(folder_path, f"{idx}.png"))
