from skimage.io import imread, imshow
from skimage.color import rgb2gray, rgb2hsv
import matplotlib.pyplot as plt
import numpy as np


# Load Image
img = imread('cones.png')[:, :, :3]
img_hsv = rgb2hsv(img)
img_gs_1c = rgb2gray(img)
# Grayscale image with 3 channels (the value is triplicated)
img_gs = ((np.stack([img_gs_1c] * 3, axis=-1) * 255)
          .astype('int').clip(0, 255))
# Plot
fig, ax = plt.subplots(1, 3, figsize=(21, 7))
ax[0].set_title("Hue Channel")
ax[0].imshow(img_hsv[:, :, 0], cmap='gray')
ax[0].set_axis_off()
ax[1].set_title("Saturation Channel")
ax[1].imshow(img_hsv[:, :, 1], cmap='gray')
ax[1].set_axis_off()
ax[2].set_title("Value Channel")
ax[2].imshow(img_hsv[:, :, 2], cmap='gray')
ax[2].set_axis_off()
plt.show()

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(img_hsv[:,:,0],cmap='hsv')
ax[0].set_title('hue')
ax[1].imshow(img_hsv[:,:,1],cmap='hsv')
ax[1].set_title('transparency')
ax[2].imshow(img_hsv[:,:,2],cmap='hsv')
ax[2].set_title('value')
fig.colorbar(imshow(img_hsv[:,:,0],cmap='hsv'))
plt.show()

# refer to hue channel (in the colorbar)
lower_mask = img_hsv[:, :, 0] > 0.6  # refer to hue channel (in the colorbar)
upper_mask = img_hsv[:, :, 0] < 0.7  # refer to transparency channel (in the colorbar)
saturation_mask = img_hsv[:, :, 1] > 0.3

mask = upper_mask * lower_mask * saturation_mask
red = img[:, :, 0] * mask
green = img[:, :, 1] * mask
blue = img[:, :, 2] * mask
cones_masked = np.dstack((red, green, blue))
plt.imshow(cones_masked)
plt.title('Segmented Cones')
plt.axis('off')
plt.show()

# refer to hue channel (in the colorbar)
lower_mask = img_hsv[:, :, 0] > 0.1  # refer to hue channel (in the colorbar)
upper_mask = img_hsv[:, :, 0] < 0.2  # refer to transparency channel (in the colorbar)
saturation_mask = img_hsv[:, :, 1] > 0.6

mask = upper_mask * lower_mask * saturation_mask
red = img[:, :, 0] * mask
green = img[:, :, 1] * mask
blue = img[:, :, 2] * mask
cones_masked = np.dstack((red, green, blue))
plt.imshow(cones_masked)
plt.title('Segmented Cones')
plt.axis('off')
plt.show()
