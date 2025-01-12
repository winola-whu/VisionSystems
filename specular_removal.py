from PIL import Image
import numpy as np
import cv2
import imageio.v3 as iio
from scipy.ndimage.morphology import binary_closing, binary_opening


def binarize_array(numpy_array, threshold=200):
    """
    Binarize a numpy array using a threshold.

    Parameters
    ----------
    numpy_array : numpy array
        The input array to binarize.
    threshold : int
        The threshold value to use for binarization.

    Returns
    -------
    numpy_array : numpy array
        The binarized array.
    """
    binarized_array = np.where(numpy_array > threshold, 255, 0).astype(np.uint8)
    return binarized_array


# Load and convert the image to grayscale
image_path = "out.png"
im = Image.open(image_path)
im = im.convert('L')
im = np.array(im)

# Binarize the image
binary_image = binarize_array(im, 200)

# Apply morphological operations
structure = np.ones((10, 10))
binary_image = binary_closing(binary_image, structure=structure).astype(np.uint8)
binary_image = binary_opening(binary_image, structure=structure).astype(np.uint8)

# Ensure the mask has the same dimensions as the original image
original_image = cv2.imread(image_path)
binary_image = cv2.resize(binary_image, (original_image.shape[1], original_image.shape[0]))

# Inpainting to remove specular highlights
# Convert the original image to BGR if it's not already
if len(original_image.shape) == 2 or original_image.shape[2] == 1:
    original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

# Perform inpainting
inpainted_image = cv2.inpaint(original_image, binary_image, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

# Save the result
output_path = "out2.png"
iio.imwrite(output_path, inpainted_image)
