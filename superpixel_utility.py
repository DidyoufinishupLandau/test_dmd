import numpy as np
from itertools import combinations
from decimal import Decimal, getcontext
import matplotlib.pyplot as plt
from scipy.special import genlaguerre
########################################################

def get_combination_sums_and_indices(input_list):
    sums_list = []
    indices_list = []

    for r in range(1, len(input_list) + 1):
        for combo_indices in combinations(range(len(input_list)), r):
            combo_sum = sum(input_list[i] for i in combo_indices)
            sums_list.append(combo_sum)
            indices_list.append(list(combo_indices))

    return sums_list, indices_list
def remove_complex_duplicates(arr,combination, precision=10):
    getcontext().prec = precision
    unique_elements = []
    unique_combination = []
    seen_elements = set()

    for i, num in enumerate(arr):
        rounded_real = round(Decimal(num.real), precision)
        rounded_imag = round(Decimal(num.imag), precision)

        if (rounded_real, rounded_imag) not in seen_elements:
            unique_elements.append(num)
            seen_elements.add((rounded_real, rounded_imag))
            unique_combination.append(combination[i])

    return unique_elements,unique_combination
def gaussian_laguerre_cartesian(l, p, x, y):
    """
    Calculate the normalized Laguerre-Gaussian beam in Cartesian coordinates.

    Parameters:
        l (int): Radial mode parameter.
        p (int): Azimuthal mode parameter.
        x (float): x-coordinate.
        y (float): y-coordinate.

    Returns:
        complex: Value of the Laguerre-Gaussian beam at given coordinates.
    """
    # Calculate the squared radial distance
    rho_squared = x ** 2 + y ** 2

    # Calculate the Laguerre polynomial term
    laguerre_term = genlaguerre(p, l)(rho_squared)

    # Calculate the angular term
    angular_term = np.exp(1j * l * np.arctan2(y, x))

    # Normalize the Laguerre-Gaussian beam
    normalized_beam = laguerre_term * angular_term

    return normalized_beam
def plot_gaussian_laguerre_cartesian(l, p, X, Y, Z):

    plt.contourf(X, Y, Z, cmap='viridis', levels=100)
    plt.colorbar(label='Intensity')
    plt.title(f'Gaussian Laguerre, l={l}, p={p}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    return None
def phase_to_superpixel(phase_matrix,pixel_phase, error=10**-2):
    """
    match the element in phase matrix with elements in the complex plan
    within given error margin.
    :param phase_matrix:
    :param error:
    :return:
    """
    superpixel_matrix = []
    error_map = []
    error_scale = np.linspace(error, 1,100)
    sum_array, combo = get_combination_sums_and_indices(pixel_phase)
    sum_array, combo = remove_complex_duplicates(sum_array, combo)
    num_row = phase_matrix.shape[0]
    num_column = phase_matrix.shape[1]
    sum_array = np.array(sum_array)
    phase_matrix = np.ravel(phase_matrix)
    for i in range(len(phase_matrix)):
        temp = abs(sum_array - phase_matrix[i])
        min = np.min(temp)
        min_indices = np.where(temp == min)[0]
        if min < error:
            superpixel_matrix.append(combo[min_indices[0]])
            error_map.append(error)
        else:
            for i in range(len(error_scale)):
                if min < error_scale[i]:
                    superpixel_matrix.append(combo[min_indices[0]])
                    error_map.append(error)
                    break
    #now map the resize the error map with each element represent as a superpixel
    #meanwhile use the fill the superpixel with masks
    def create_array_with_indices(indices, shape=(4,4)):
        array = np.zeros(shape, dtype=int)
        for index in indices:
            row = int(np.floor(index/shape[0]))
            column = np.mod(index,shape[1])
            array[row][column] = 255
        return array

    temp_superpixel = np.zeros((4, 0))
    count = 0

    two_dimension_superpixel_matrix = np.zeros((0,num_column*4))
    for i in range(len(superpixel_matrix)):

        temp_superpixel = np.hstack((temp_superpixel,create_array_with_indices(superpixel_matrix[i])))
        count+=1
        if count==num_column:
            two_dimension_superpixel_matrix = np.vstack((two_dimension_superpixel_matrix, temp_superpixel))
            count=0
            temp_superpixel = np.zeros((4,0))

    return two_dimension_superpixel_matrix
# Example usage:
