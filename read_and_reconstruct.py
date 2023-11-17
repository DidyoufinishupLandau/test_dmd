import os
import time
import numpy as np
import generate_pattern
import matplotlib.pyplot as plt
from skimage import transform
from matplotlib.colors import LinearSegmentedColormap
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
def get_data(count, pixel, group):
    data_one = np.array([])
    data_two = np.array([])
    for i in range(count):
        temp_data_one = np.genfromtxt(f"{pixel}_{group}_one_data_{i+1}.csv",delimiter=',')
        temp_data_one = temp_data_one[0:4096]
        temp_data_two = np.genfromtxt(f'{pixel}_{group}_two_data_{i+1}.csv',delimiter=',')
        temp_data_two = temp_data_two[0:4096]
        data_one = np.hstack((data_one, temp_data_one))
        data_two = np.hstack((data_two, temp_data_two))
    print(len(data_one))
    return data_one, data_two
def combine_data(intensity, size):
    positive_pattern, negative_pattern = generate_pattern.DmdPattern('hadamard', size, size).execute()
    image = []
    i=0
    length_pattern = len(positive_pattern)
    intensity_length = len(intensity)
    while i<intensity_length:
        image.append((intensity[i]*(positive_pattern[0]-negative_pattern[0])).astype(np.float16))
        i+=1
        #print(positive_pattern[0])
        del positive_pattern[0]
        del negative_pattern[0]
        #print(positive_pattern[0])
    image = np.sum(np.array(image), axis=0)/length_pattern
    return  image
def plot_pixel(image_matrix):
    cdict = {'red': ((0.0, 0.0, 0.0),
                     (1.0, 1.0, 1.0)),
             'green': ((0.0, 0.0, 0.0),
                       (1.0, 0.0, 0.0)),
             'blue': ((0.0, 0.0, 0.0),
                      (1.0, 0.0, 0.0))}

    red_cmap = LinearSegmentedColormap('RedMap', segmentdata=cdict, N=256)

    plt.imshow(image_matrix, cmap="gray")
    plt.show()

def smooth_image(matrix, kernel_size=3):
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size should be odd")

    rows = len(matrix)
    cols = len(matrix[0])

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

def replace_largest(arr):
    ave = np.average(arr)
    largest_value = np.max(arr)
    arr[arr == largest_value] = ave
    return arr
def replace_smallest(arr):
    largest_value = np.min(arr)
    arr[arr == largest_value] = np.average(arr)
    return arr
def remove_row(arr, row_to_remove):
    return np.delete(arr, row_to_remove, axis=0)

def remove_tw(arr):
    ave = np.average(arr)
    arr[0][0] = ave
    arr[0][32] = ave
    arr[32][0] = ave
    arr [32][32] = ave
    return arr
def remove_on(arr):
    ave = np.average(arr)
    arr[0][0] = ave
    arr[0][64] = ave
    arr[64][0] = ave
    arr [64][64] = ave
    return arr

def embed(pattern):
    height,width = pattern[0].shape
    DMD_height = 1140
    DMD_width = 912

    height_start = int(DMD_height/2) - int(height/2)
    height_end = int(DMD_height/2)+int(height/2)
    width_start = int(DMD_width/2)-int(height/2)
    width_end = int(DMD_width/2)+int(height/2)
    new_pattern = []
    def inner_loop(two_dimension_pattern):
        one = (np.ones((DMD_height, DMD_width)) * 128).astype(np.uint8)
        one[height_start:height_end, width_start: width_end] = two_dimension_pattern * 255
        return one
    return list(map(inner_loop, pattern))

def fourier_transform(image):
    f = np.fft.fft2(image)

    # Shift the zero frequency component (DC component) to the center
    fshift = np.fft.fftshift(f)

    # Calculate the magnitude spectrum from the complex numbers
    fourier_spectrum = 20 * np.log(np.abs(fshift))
    ifft = np.fft.ifft2(image)
    ifshit = np.fft.ifftshift(ifft)
    ifshit = np.abs(ifshit)
    ave = np.average(ifshit)
    ifshit[64][64] = ave
    return  fourier_spectrum, ifshit
def main():

    data_one, data_two = get_data(2, 128, 12)
    image = combine_data((data_one-data_two)/(data_one+data_two), size=128)
    fourier,inver = fourier_transform(image)
    image = remove_on(image)
    #fourier = remove_tw(fourier)
    plot_pixel(image)
    plot_pixel(inver)
main()