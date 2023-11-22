import generate_pattern
import numpy as np
from skimage import transform
import matplotlib.pyplot as plt
def rescale(input_image: np.array) -> np.array:
    """
    resize any size image into 1140 * 912 image.
    :param input_image: single mask
    :return:
    2D array: single mask
    """
    rs = transform.resize(input_image, (512, 512), order=0, anti_aliasing=False)
    return rs
def plot_pixel(image_matrix):
    plt.imshow(image_matrix, cmap="gray")
    plt.show()