import os
import time
import numpy as np
import generate_pattern
import matplotlib.pyplot as plt
from skimage import transform
count = 4
def get_calibration():
    data_one = np.array([])
    data_two = np.array([])
    temp_data_one = np.genfromtxt(f"one_data_{0}.csv",delimiter=',')
    temp_data_two = np.genfromtxt(f'two_data_{0}.csv',delimiter=',')
    data_one = np.hstack((data_one, temp_data_one))
    data_two = np.hstack((data_two, temp_data_two))
    average_deviation = np.sum(data_one-data_two)/len(data_one)
    deviation_std = np.std((data_one-data_two), axis=0)
    return average_deviation, deviation_std
def get_data():
    data_one = np.array([])
    data_two = np.array([])
    for i in range(count):
        temp_data_one = np.genfromtxt(f"one_data_{i+1}.csv",delimiter=',')
        temp_data_two = np.genfromtxt(f'two_data_{i+1}.csv',delimiter=',')
        data_one = np.hstack((data_one, temp_data_one))
        data_two = np.hstack((data_two, temp_data_two))
    print(len(data_one))
    return data_one, data_two
def combine_data(pattern, intensity):
    image = []
    for i in range(1, len(pattern)):
        temp_pattern = pattern[i]
        reverse_pattern = (temp_pattern==0).astype(int)*-1
        image.append(intensity[i]*(temp_pattern+reverse_pattern))
    image = np.sum(np.array(image), axis=0)/len(pattern)
    return  image
def plot_pixel(image_matrix):
    plt.imshow(image_matrix, cmap="gray")
    plt.show()

def smooth_image(matrix, kernel_size=7):
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size should be odd")

    # Get the dimensions of the matrix
    rows = len(matrix)
    cols = len(matrix[0])

    # Prepare the output smoothed image
    smoothed = [[0 for _ in range(cols)] for _ in range(rows)]

    offset = kernel_size // 2

    for i in range(rows):
        for j in range(cols):
            # Calculate mean for the neighborhood
            sum_val = 0
            count = 0
            for k in range(-offset, offset + 1):
                for l in range(-offset, offset + 1):
                    if 0 <= i + k < rows and 0 <= j + l < cols:
                        sum_val += matrix[i + k][j + l]
                        count += 1

            smoothed[i][j] = sum_val / count

    return smoothed

def replace_largest(arr, replacement=0):
    largest_value = np.max(arr)
    arr[arr == largest_value] = replacement
    return arr
def replace_smallest(arr, replacement=0):
    largest_value = np.min(arr)
    arr[arr == largest_value] = replacement
    return arr
data_one, data_two = get_data()
print(data_one)
print(data_two)
gp = generate_pattern.DmdPattern('hadamard', 64,64)
pattern = gp.execute(two_dimension=True)


image = combine_data(pattern, (data_one-data_two)/(data_one+data_two))
image = smooth_image(image)
image = smooth_image(image)
image = smooth_image(image)
image = smooth_image(image)
plot_pixel(image)