import os
import torch
import cv2
import numpy as np
from torchvision.utils import save_image
import torchvision.transforms as T

# scene = "garage"

# data_folder = f"/run/determined/workdir/home/data/SCINeRF-data/input_data/{scene}"
data_folder = "/run/determined/workdir/home/gsplat/examples/data/sci_nerf/vggsfm/real/garage_colored"

mask_path = os.path.join(data_folder, "mask.npy")
meas_path = os.path.join(data_folder, "meas.npy")

masks_folder_path = os.path.join(data_folder, "masks")

if os.path.isdir(masks_folder_path):
    print("Masks directory exists")
else:
    print("Masks directory doesn't exists, thus create")
    os.mkdir(masks_folder_path)

masks = torch.Tensor(np.load(mask_path))
meas = torch.Tensor(np.load(meas_path))

transform = T.ToPILImage(mode="L")

for i, mask in enumerate(masks):
    img = transform(mask)
    img.save(os.path.join(masks_folder_path, f"{i}.png"))
    # cv2.imwrite(os.path.join(masks_folder_path, f"{i:03d}.png"), mask.numpy())
    # save_image(mask, os.path.join(masks_folder_path, f"{i:03d}.png"))
