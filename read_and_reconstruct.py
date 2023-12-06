import os
import time
import numpy as np
import generate_pattern
import matplotlib.pyplot as plt
from skimage import transform
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
from simulation_DMD import PSNR
import cv2

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
        temp_data_two = np.genfromtxt(f'{pixel}_{group}_two_data_{i+1}.csv',delimiter=',')
        data_one = np.hstack((data_one, temp_data_one))
        data_two = np.hstack((data_two, temp_data_two))
    print(len(data_one))
    return data_one, data_two

def combine_data(intensity, size, length=1, name="hadamard"):
    positive_pattern, negative_pattern = generate_pattern.DmdPattern(name, size, size).execute(length)
    image = []
    pattern = np.array(positive_pattern)-np.array(negative_pattern)
    length_pattern = len(positive_pattern)
    intensity_length = len(intensity)
    for i in range(len(intensity)):
        image.append((intensity[i]*pattern[i]).astype(np.float64))
    image = np.sum(np.array(image), axis=0)/length_pattern
    return  image

def combine_data_batch(intensity, positive, negative, length=1, name="hadamard"):
    image = []
    pattern = np.array(positive)-np.array(negative)
    length_pattern = len(positive)
    intensity_length = len(intensity)
    print("here")
    for i in range(int(len(positive)*length)):
        image.append((intensity[i]*pattern[i]).astype(np.float64))
    image = np.sum(np.array(image), axis=0)/length_pattern
    return  image
def plot_pixel(image_matrix):
    # Create a figure and an axes instance
    fig, ax = plt.subplots()

    # Display the image
    im = ax.imshow(image_matrix, cmap='hot')

    # Add title with PSNR value (assuming mask_length is a list and i is the index)
    ax.set_title(f"Multimodal")

    # Add a colorbar
    fig.colorbar(im, ax=ax)

    # Show the plot
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
    arr[0][1] = ave
    arr[0][2] = ave
    arr[0][3] = ave
    arr[0][31] = ave
    arr[0][32] = ave
    arr[0][33] = ave
    arr[0][34] = ave
    arr[32][32] = ave
    arr[32][34] = ave
    arr[32][0] = ave
    arr[32][1] = ave
    arr[32][2] = ave
    arr[32][3] = ave
    arr = replace_smallest(arr)

    return arr
def remove_on(arr):
    ave = np.average(arr)
    arr[0][0] = ave
    arr[0][64] = ave
    arr[64][0] = ave
    arr [64][64] = ave
    arr [64][1] = ave
    arr [64][65] = ave
    arr [64][66] = ave
    return arr
def rescale_2d_array(arr):
    # Normalizing the array to [0, 1]
    min_val = arr.min()
    max_val = arr.max()
    normalized_arr = (arr - min_val) / (max_val - min_val)

    # Rescaling to [-255, 255]
    rescaled_arr = (normalized_arr * 255)

    return rescaled_arr

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
    #ifshit[64][64] = ave
    return  fourier_spectrum, ifshit
def average_t(arr):
    ave = np.average(arr)
    arr[0][0] = ave
    arr[0][8] = ave
    arr[8][8] = ave
    arr[8][0] = ave
    return  arr
def main_sixteen():

    data_one, data_two = get_data(1, 16, 1)
    image = combine_data((data_one-data_two)/(data_one+data_two), size=16)
   # fourier,inver = fourier_transform(image)
    #image = remove_tw(image)
    image = average_t(image)
    #fourier = remove_tw(fourier)
    #image= smooth_image(image)
    plot_pixel(np.log(image))
    #plot_pixel(inver)
def main_thirty_two():
    data_one, data_two = get_data(1, 32, 1)
    one_frame_size = 1024
    for i in range(1):
        print(one_frame_size*i)
        temp_data_one = data_one[one_frame_size*i:one_frame_size*(i+1)]
        temp_data_two = data_two[one_frame_size*i:one_frame_size*(i+1)]
        image = combine_data((temp_data_two-temp_data_one), size=32)
        plot_pixel(image)
def main_six_four(name):
    data_one, data_two = get_data(1, 64, 6)
    one_frame_size = 4096
    for i in range(1):
        print(one_frame_size*i)
        temp_data_one = data_one[one_frame_size*i:one_frame_size*(i+1)]
        temp_data_two = data_two[one_frame_size*i:one_frame_size*(i+1)]
        image = combine_data((temp_data_two-temp_data_one), size=64, name=name)
        image = remove_tw(image)
        print('f')
        # fourier = remove_tw(fourier)
        #image= smooth_image(image)
        plot_pixel(image)
def main_one_two_eight():
    data_one, data_two = get_data(4, 128, 12)
    image = combine_data((data_two - data_one) , size=128)
    image = remove_on(image)
    image[64] = np.average(image)
    plot_pixel(np.log(image))

def main_one_two_five():
    data_one, data_two = get_data(4, 256, 1)
    image = combine_data((data_two - data_one)/(data_two+data_one), size=256, length=1)
    fourier, inver = fourier_transform(image)
    print(image)
    # image = remove_tw(image)
    # image = remove_tw(image)
    # fourier = remove_tw(fourier)
    # image= smooth_image(image)
    plot_pixel(image)
    plot_pixel(inver)

for i in range(40):
    try:
        data_one, data_two = get_data(4, 128, i+1)
        combined_data = combine_data_batch((data_one-data_two), positive, negative)
        rescaled_data = rescale_2d_array(combined_data)

        plot_pixel(rescaled_data)
    except FileNotFoundError:
        print("not found pass")