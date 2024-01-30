import numpy as np
import generate_pattern
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class ReadAndReconstruction():
    def __init__(self, image_size):
        self.image_size = image_size
        positive_pattern, negative_pattern = generate_pattern.DmdPattern('hadamard', image_size, image_size).execute()
        self.pattern = np.array(positive_pattern)-np.array(negative_pattern)
        self.pattern_length = self.pattern.shape[0]
        self.image = 0
    def get_data(self, group):
        data_one = np.array([])
        data_two = np.array([])
        for i in range(4**(int(self.image_size/64-1))):
            temp_data_one = np.genfromtxt(f"{self.image_size}_{group}_one_data_{i + 1}.csv", delimiter=',')
            temp_data_one = temp_data_one[0:4096]
            temp_data_two = np.genfromtxt(f'{self.image_size}_{group}_two_data_{i + 1}.csv', delimiter=',')
            temp_data_two = temp_data_two[0:4096]
            data_one = np.hstack((data_one, temp_data_one))
            data_two = np.hstack((data_two, temp_data_two))
        return data_two - data_one
    def get_image(self, group):
        data = self.get_data(group)
        data = np.pad(data,(0,self.image_size**2 - data.shape[0]))
        image = []
        for i in range(len(data)):
            image.append((data[i] * self.pattern[i]))
        image = np.sum(np.array(image), axis=0) / self.pattern_length
        self.image = image
        return image
    def plot(self):
        cdict = {'red': ((0.0, 0.0, 0.0),
                         (1.0, 1.0, 1.0)),
                 'green': ((0.0, 0.0, 0.0),
                           (1.0, 0.0, 0.0)),
                 'blue': ((0.0, 0.0, 0.0),
                          (1.0, 0.0, 0.0))}

        red_cmap = LinearSegmentedColormap('RedMap', segmentdata=cdict, N=256)

        plt.imshow(self.image, cmap="gray")
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
RR = ReadAndReconstruction(32)
RR.get_image(4)
RR.plot()