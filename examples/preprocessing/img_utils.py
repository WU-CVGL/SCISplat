import os
import math
import torch
from torchvision.utils import save_image
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import griddata

def interpolate_img_with_mask(img, mask):
    """interpolate image with neareat neighbour
    Args:
        img: torch.Tensor (3, H, W)
        mask: torch.Tensor (H, W)
    
    Returns:
        interpolated_img: torch.Tensor (3, H, W)

    """
    # interpolate it with nearest neighbour and observe if noise reduced significantly 
    img = img.permute(1,2,0) # (H, W, 3)
    H, W, _ = img.shape

    x, y = torch.where(mask == 1) #
    points = np.stack((x,y)).T
    decoded_img_values = img[points[:, 0], points[:, 1], :]
    grid_x, grid_y = np.mgrid[0:H, 0:W]
    grid_z0 = griddata(points, decoded_img_values, (grid_x, grid_y), method='nearest')

    interpolated_img = torch.tensor(grid_z0).permute(2,0,1)

    return interpolated_img

def masked_median_filter(masked_img: torch.Tensor, mask: torch.Tensor, kernel_size: int):
    """apply median filtering on masked image

    Args:
        masked_img: torch.Tensor (3, H, W)
        mask: torch.Tensor (H, W)
        kernel_size: int 

    Return:
        filtered_img: torch.Tensor (3, H, W)
    """
    filtered_img = masked_img.clone() # deepcopy?
    H, W = mask.shape
    edge_offset = kernel_size // 2
    indices_h, indices_w = H - 2 * edge_offset, W - 2 * edge_offset
    # convolve around the mask
    for i in range(indices_h):
        for j in range(indices_w):
            idx_H, idx_W = i + edge_offset, j + edge_offset
            # check if pixel is masked out at this position
            if mask[idx_H, idx_W] == 1:
                neighbors_mask = mask[idx_H - edge_offset:idx_H + edge_offset + 1, idx_W - edge_offset:idx_W + edge_offset + 1]
                neighbors = masked_img[:, idx_H - edge_offset:idx_H + edge_offset + 1, idx_W - edge_offset:idx_W + edge_offset + 1]
                # only preserve unmasked neighbors
                neighbors = neighbors[:, neighbors_mask.bool()] # (3, *)
                if neighbors.shape[1] > 1: # Ensure there are unmasked neighbors
                    neighbors_median = torch.median(neighbors, 1)[0]
                    filtered_img[:, idx_H, idx_W] = neighbors_median

    return filtered_img

def color_distance(rgb_1, rgb_2):
    rgb_1 = rgb_1 * 255
    rgb_2 = rgb_2 * 255
    r_mean = (rgb_1[0] + rgb_2[0]) / 2
    r_diff = rgb_1[0] - rgb_2[0]
    g_diff = rgb_1[1] - rgb_2[1]
    b_diff = rgb_1[2] - rgb_2[2]

    return math.sqrt((2+r_mean/256)*(r_diff**2) + 4*(g_diff**2) + (2+(255-r_mean)/256) * (b_diff ** 2))

def find_outliers_wrt_median(masked_img: torch.Tensor, mask: torch.Tensor, kernel_size: int, threshold=100):
    """find pixels that's vastly different from its neighbors 
    
    Args:
        masked_img: torch.Tensor (3, H, W)
        mask: torch.Tensor (H, W)
        kernel_size: int 

    Return:
        indices: torch.Tensor (N, 2) pixel locations in masked_img
    
    """
    # TODO: speed up this implementation with vectorization
    filtered_img = masked_img.clone() # deepcopy?
    H, W = mask.shape
    edge_offset = kernel_size // 2
    indices_h, indices_w = H - 2 * edge_offset, W - 2 * edge_offset
    
    color_dist_all = []
    indices = [] 
    # convolve around the mask
    for i in range(indices_h):
        for j in range(indices_w):
            idx_H, idx_W = i + edge_offset, j + edge_offset
            # check if pixel is masked out at this position
            if mask[idx_H, idx_W] == 1:
                neighbors_mask = mask[idx_H - edge_offset:idx_H + edge_offset + 1, idx_W - edge_offset:idx_W + edge_offset + 1]
                neighbors = masked_img[:, idx_H - edge_offset:idx_H + edge_offset + 1, idx_W - edge_offset:idx_W + edge_offset + 1]
                # only preserve unmasked neighbors
                neighbors_valid = neighbors[:, neighbors_mask.bool()] # (3, *)
                if neighbors_valid.shape[1] > 1: # Ensure there are unmasked neighbors
                    neighbors_median = torch.median(neighbors_valid, 1)[0]
                    # filtered_img[:, idx_H, idx_W] = neighbors_median

                    # check if the current pixel deviates vastly from the median
                    color_dist = color_distance(filtered_img[:, idx_H, idx_W], neighbors_median)
                    color_dist_all.append(color_dist)
                    if color_dist > threshold:
                        indices.append(torch.tensor([idx_H, idx_W]))

    indices = torch.stack(indices)
    print(f'Tick out {indices.shape[0]} pixels.')
    filtered_img[:, indices[:, 0], indices[:, 1]] = torch.tensor([1.,0,0]).unsqueeze(-1)
    mask[indices[:, 0], indices[:, 1]] = 0

    return indices, mask

def find_outliers_wrt_median_fast(masked_img, mask, kernel_size, threshold=100):

    raise NotImplementedError

    return indices

if __name__ == "__main__":

    # read in a decoded image and its mask
    scene = 'airplants'

    img_path = f'/run/determined/workdir/home/data/SCINeRF-data/input_data/{scene}/meas.npy'
    mask_path = f'/run/determined/workdir/home/data/SCINeRF-data/input_data/{scene}/mask.npy'

    # set the kernel size (odd number: 5x5, 7x7, 9x9 ...)
    kernel_size = 7
    threshold = 100

    comp_img = torch.Tensor(np.load(img_path))

    H, W, _ = comp_img.shape

    masks = torch.Tensor(np.load(mask_path))
 
    # compressive image normalization
    masks_sum = masks.sum(0)
    comp_img = comp_img / masks_sum.unsqueeze(-1)

    decoded_imgs = []

    for mask in masks:
        decoded_img = comp_img * mask.unsqueeze(-1)

        decoded_imgs.append(decoded_img.permute(2,0,1)) 


    interpolated_imgs = []
    # get the filtered decoded image
    for decoded_img, mask in zip(decoded_imgs, masks):
        
        # NOTE: this function is now slow as fuck, need to boost it up in the future
        indices, mask = find_outliers_wrt_median(decoded_img, mask, kernel_size=kernel_size, threshold=threshold) # (3, H, W)

        interpolated_img = interpolate_img_with_mask(decoded_img, mask)
        interpolated_imgs.append(interpolated_img)

    filter_comment = f'median-tick-{kernel_size}-{threshold}'
    folder_path = f'/run/determined/workdir/home/SCI-Gaussians/test/decoded_imgs_{scene}_nearest_{filter_comment}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created at {folder_path}")
    else:
        print(f"Folder already exists at {folder_path}")


    # save images ina folder
    for idx, interpolated_img in enumerate(interpolated_imgs):
        
        save_image(interpolated_img, os.path.join(folder_path, f'frame_{idx}.png'))