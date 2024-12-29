import numpy as np
from PIL import Image

def npy_to_png(npy_file, png_file):
    # Load the .npy file
    data = np.load(npy_file)

    # Ensure the data is scaled to the range 0-255 and converted to uint8
    # If data is already in uint8 and within the correct range, you can skip this step
    if data.dtype != np.uint8:
        # Normalize the data to 0-255
        data = (255 * (data - data.min()) / (data.max() - data.min())).astype(np.uint8)

    # Convert the NumPy array to an image
    image = Image.fromarray(data)

    # Save the image as a PNG file
    image.save(png_file)
    print(f"Image saved to {png_file}")

# Example usage:
npy_to_png('/run/determined/workdir/home/gsplat/examples/data/sci_nerf/vggsfm/real/garage_qf1_pts2048/meas.npy', 'meas.png')
