from skimage.io import imread, imshow
from skimage.color import rgb2gray, rgb2hsv
import matplotlib.pyplot as plt
import numpy as np
from timeit import timeit


# Load Image
img = imread('hard_case_2.png')[:, :, :3]
img_hsv = rgb2hsv(img)


def segment(lower, upper, saturation, img_hsv):
    lower_mask = img_hsv[:, :, 0] > lower  # refer to hue channel (in the colorbar)
    upper_mask = img_hsv[:, :, 0] < upper  # refer to transparency channel (in the colorbar)
    saturation_mask = img_hsv[:, :, 1] > saturation

    mask = upper_mask * lower_mask * saturation_mask
    red = img[:, :, 0] * mask
    green = img[:, :, 1] * mask
    blue = img[:, :, 2] * mask
    cones_masked = np.dstack((red, green, blue))
    plt.imshow(img_hsv[:, :, 0], cmap='hsv')
    plt.colorbar()

    return cones_masked


iter = 100
#total_time = timeit("segment(0.6, 0.7, 0.3, img_hsv)", number=iter, globals=globals())
#print(f"Average time is {total_time / iter:.2f} seconds")
plt.imshow(img_hsv)
plt.colorbar()
plt.show()

plt.imshow(segment(0.6, 0.7, 0.3, img_hsv))
plt.title('Segmented Blue Cones')
plt.axis('off')
plt.show()

plt.imshow(segment(0.1, 0.2, 0.6, img_hsv))
plt.title('Segmented Yellow Cones')
plt.axis('off')
plt.show()

plt.imshow(segment(0.0, 0.1, 0.5, img_hsv))
plt.title('Segmented Orange Cones')
plt.axis('off')
plt.show()