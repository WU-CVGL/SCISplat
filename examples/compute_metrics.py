import os
import re
from PIL import Image
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms import ToTensor


# Function to extract numbers from filenames
def extract_number(filename):
    match = re.search(r"\d+", filename)
    return int(match.group()) if match else -1


# Function to load and sort images
def load_and_sort_images_from_folder(folder):
    images = []
    filenames = sorted(
        [f for f in os.listdir(folder) if f.endswith(".png")], key=extract_number
    )

    for filename in filenames:
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert("RGB")  # Convert to RGB
        img_tensor = ToTensor()(img)  # Convert to torch tensor
        images.append(img_tensor)

    return torch.stack(images)


if __name__ == "__main__":
    # initialize metrics
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    psnr = PeakSignalNoiseRatio(data_range=1.0)
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    # path of rendered images and GT
    rendered_path = (
        "/run/determined/workdir/home/gsplat/examples/results/scinerf/img_test_200000"
    )
    GT_path = "/run/determined/workdir/home/gsplat/examples/data/sci_nerf/vggsfm/kun_decoded_qf1/images"

    # read in rendered images
    rendered_imgs = load_and_sort_images_from_folder(rendered_path)  # (N, 3, H, W)

    # read in GT images
    GT_imgs = load_and_sort_images_from_folder(GT_path)  # (N, 3, H, W)

    mean_psnr = psnr(rendered_imgs, GT_imgs)
    mean_ssim = ssim(rendered_imgs, GT_imgs)
    mean_lpips = lpips(rendered_imgs, GT_imgs)

    print(f"Mean PSNR: {mean_psnr}, Mean SSIM: {mean_ssim}, Mean LPIPS: {mean_lpips}.")
