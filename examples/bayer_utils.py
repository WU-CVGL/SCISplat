import torch


def rgb_to_bayer_rggb(rgbs):
    N, H, W, _ = rgbs.shape

    # Initialize Bayer pattern image
    bayer_rggb = torch.zeros((N, H, W), dtype=rgbs.dtype, device=rgbs.device)

    # Extract channels
    r = rgbs[:, :, :, 0]
    g = rgbs[:, :, :, 1]
    b = rgbs[:, :, :, 2]

    # RGGB pattern
    bayer_rggb[:, 0:H:2, 0:W:2] = r[:, 0:H:2, 0:W:2]  # R
    bayer_rggb[:, 0:H:2, 1:W:2] = g[:, 0:H:2, 1:W:2]  # G
    bayer_rggb[:, 1:H:2, 0:W:2] = g[:, 1:H:2, 0:W:2]  # G
    bayer_rggb[:, 1:H:2, 1:W:2] = b[:, 1:H:2, 1:W:2]  # B

    return bayer_rggb


def rgb_to_bayer_grbg(rgbs):
    N, H, W, _ = rgbs.shape

    # Initialize Bayer pattern image
    bayer_grbg = torch.zeros((N, H, W), dtype=rgbs.dtype)

    # Extract channels
    r = rgbs[:, :, :, 0]
    g = rgbs[:, :, :, 1]
    b = rgbs[:, :, :, 2]

    # GRBG pattern
    bayer_grbg[:, 0:H:2, 0:W:2] = g[:, 0:H:2, 0:W:2]  # G
    bayer_grbg[:, 0:H:2, 1:W:2] = r[:, 0:H:2, 1:W:2]  # R
    bayer_grbg[:, 1:H:2, 0:W:2] = b[:, 1:H:2, 0:W:2]  # B
    bayer_grbg[:, 1:H:2, 1:W:2] = g[:, 1:H:2, 1:W:2]  # G

    return bayer_grbg
