import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from array_utils.math import rescale


def plot_img(img):
    if isinstance(img, str):
        img = mpimg.imread(img)
    else:
        img[np.isnan(img)] = 0.
        img = rescale(img, 0, 255).astype(np.uint8)
        if img.shape[0] == 3:
            img = img.swapaxes(0, 2)
    imgplot = plt.imshow(img)
    plt.show()
