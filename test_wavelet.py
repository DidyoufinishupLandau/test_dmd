import numpy as np
import pywt
import matplotlib.pyplot as plt
from simulation_DMD import random_image_with_shapes
image = random_image_with_shapes(128,128,5)

def wavelet_transform_level_one(matrix):
    # Apply single level Discrete Wavelet Transform
    coeffs = pywt.dwt2(matrix, 'db1')  # 'db1' refers to the Daubechies wavelet with one vanishing moment.
    cA, (cH, cV, cD) = coeffs  # cA: approximation, cH: horizontal details, cV: vertical details, cD: diagonal details

    return cA, cH, cV, cD
def inverse_wavelet_trasform(cA, cH, cV, cD):
    reconstructed_data_2D = pywt.idwt2((cA, (cH, cV, cD)), 'db1')
    return reconstructed_data_2D

"""
# Example matrix (for demonstration purposes)
matrix = image

cA, cH, cV, cD = wavelet_transform_level_one(matrix)
# If you wish to visualize the coefficients
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.imshow(cA, cmap='gray')
plt.title("Approximation Coefficients")

plt.subplot(2, 2, 2)
plt.imshow(cH, cmap='gray')
plt.title("Horizontal Details")

plt.subplot(2, 2, 3)
plt.imshow(cV, cmap='gray')
plt.title("Vertical Details")

plt.subplot(2, 2, 4)
plt.imshow(cD, cmap='gray')
plt.title("Diagonal Details")

plt.tight_layout()
plt.show()
cA, cH, cV, cD = wavelet_transform_level_one(cA)
# If you wish to visualize the coefficients
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.imshow(cA, cmap='gray')
plt.title("Approximation Coefficients")

plt.subplot(2, 2, 2)
plt.imshow(cH, cmap='gray')
plt.title("Horizontal Details")

plt.subplot(2, 2, 3)
plt.imshow(cV, cmap='gray')
plt.title("Vertical Details")

plt.subplot(2, 2, 4)
plt.imshow(cD, cmap='gray')
plt.title("Diagonal Details")

plt.tight_layout()
plt.show()

plt.tight_layout()
plt.show()
cA, cH, cV, cD = wavelet_transform_level_one(cA)
# If you wish to visualize the coefficients
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.imshow(cA, cmap='gray')
plt.title("Approximation Coefficients")

plt.subplot(2, 2, 2)
plt.imshow(cH, cmap='gray')
plt.title("Horizontal Details")

plt.subplot(2, 2, 3)
plt.imshow(cV, cmap='gray')
plt.title("Vertical Details")

plt.subplot(2, 2, 4)
plt.imshow(cD, cmap='gray')
plt.title("Diagonal Details")

plt.tight_layout()
plt.show()
reconstructed_data_2D = inverse_wavelet_trasform(cA, cH, cV, cD)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

cax1 = ax[0].matshow(matrix, cmap='gray')
plt.colorbar(cax1, ax=ax[0])
ax[0].set_title("Original 2D Data")

cax2 = ax[1].matshow(reconstructed_data_2D, cmap='gray')
plt.colorbar(cax2, ax=ax[1])
ax[1].set_title("Reconstructed 2D Data")

plt.tight_layout()
plt.show()"""