#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 10:59:20 2023

@author: tabithajohn
"""

import numpy as np
import matplotlib.pyplot as plt
from generate_pattern import DmdPattern
from PIL import Image, ImageDraw


def random_image_with_shapes(width, height, num_shapes=5):
    # Create a black background (2D array of zeros)
    background = np.zeros((width, height), dtype=np.uint8)

    # Convert the 2D numpy array to a PIL Image
    image = Image.fromarray(background, mode='L')

    # Create a drawing context to draw shapes on the image
    draw = ImageDraw.Draw(image)

    # Randomly draw shapes
    for _ in range(num_shapes):
        # Determine shape type (either rectangle or ellipse)
        shape_type = np.random.choice(["rectangle", "ellipse"])

        # Randomly determine top-left and bottom-right points for the shape
        x1, y1 = np.random.randint(0, width - 30), np.random.randint(0, height - 30)
        x2, y2 = x1 + np.random.randint(20, 30), y1 + np.random.randint(20, 30)

        # Randomly determine the grayscale color of the shape (between 1 and 255 to ensure it's not black)
        color = np.random.randint(1, 256)

        if shape_type == "rectangle":
            draw.rectangle([x1, y1, x2, y2], fill=color)
        else:
            draw.ellipse([x1, y1, x2, y2], fill=color)

    # Convert the grayscale image to a 2D numpy array and return
    return np.asarray(image)


def plot_pixel(image_matrix):
    plt.imshow(image_matrix, cmap="gray")
    plt.show()


class SimulationCompressingImage:
    def __init__(self, simulation_image, light_intensity, width, height):
        self.light_intensity = light_intensity
        self.width = width
        self.height = height
        self.num_pixel = width * height
        self.simulation_image = simulation_image


    def simulate_hadmard_pattern(self, sampling_rate=1, noise_rate=0):
        DP = DmdPattern("hadamard", self.width, self.height)
        mask_matrix_one = DP.execute()
        image = []
        for i in range(int(len(mask_matrix_one) * sampling_rate)):
            mask = mask_matrix_one[i]
            reverse_mask = (mask == 0).astype(int)

            fractional_signal = np.sum((mask * self.simulation_image) / 255) / self.num_pixel
            reverse_fractional_signal = np.sum((reverse_mask * self.simulation_image) / 255) / self.num_pixel

            photo_diode_signal = self.light_intensity * fractional_signal
            photo_diode_reverse_signal = self.light_intensity * reverse_fractional_signal

            # signal_noise = np.random.randint(-100,100,[self.width, self.height])/100*noise_rate*self.light_intensity
            signal_noise = np.random.randint(0, 100) / 100 * noise_rate * self.light_intensity
            reverse_signal_nose = np.random.randint(0, 100) / 100 * noise_rate * self.light_intensity
            signal = photo_diode_signal + signal_noise
            reverse_signal = photo_diode_reverse_signal + reverse_signal_nose
            image.append((signal - reverse_signal) * (mask - reverse_mask))
            # image.append(signal * mask + reverse_signal * reverse_mask)
        image = np.sum(np.array(image), axis=0) / (int(len(mask_matrix_one) * sampling_rate))
        image = image + abs(np.min(image))
        return image

    def simulate_fourier_pattern(self, sampling_percentage, sampling_rate):
        image = np.fft.fftshift(np.fft.fft2(self.simulation_image))  # simulate image in fourier plane

        sampling_area = round(self.width * self.height * sampling_percentage)
        length = round(np.sqrt(sampling_area) / 2)
        x1 = round((self.width / 2) - length)
        x2 = round((self.width / 2) + length)
        y1 = round((self.height / 2) - length)
        y2 = round((self.height / 2) + length)

        # mask
        image[:(self.height), :(x1)] = 1
        image[:(y1), :(self.width)] = 1
        image[:(self.height), (x2):] = 1
        image[(y2):, :(self.width)] = 1

        plot_pixel(np.log(abs(image)))

        # hadamard sampling of selected area
        masked_image = SimulationCompressingImage(image, 1, self.width, self.height)
        hadamard_sampled = masked_image.simulate_hadmard_pattern(sampling_rate, 0.0)

        reverse_fourier = np.fft.ifft2(hadamard_sampled)

        return reverse_fourier


image = random_image_with_shapes(128, 128, 4)
plot_pixel(image)

fourier_transform = np.fft.fftshift(np.fft.fftn(image))
plot_pixel(np.log(abs(fourier_transform)))

f = SimulationCompressingImage(image, 1, 128, 128)
simulated_image = np.log(abs(f.simulate_fourier_pattern(1, 1)))
plot_pixel(simulated_image)
