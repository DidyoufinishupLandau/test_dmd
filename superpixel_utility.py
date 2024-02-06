import numpy as np
from itertools import combinations
from decimal import Decimal, getcontext
import matplotlib.pyplot as plt
from scipy.special import genlaguerre
from DMD_driver import DMD_driver
import time
from simulation_DMD import plot_pixel
########################################################
class SuperpixelPattern:
    def __init__(self, superpixel_size=4):
        self.x = np.e**(np.arange(0,superpixel_size**2,1)*2*np.pi/superpixel_size**2*1j)/superpixel_size**2
    def superpixel_pattern(self, phase_matrix):
        target_matrix = phase_to_superpixel(phase_matrix, self.x)
        return target_matrix[:, :, np.newaxis].astype(np.uint8)

    def plane_wave(self, phase):
        return_array = np.exp(1j*phase)*np.ones((228, 286))
        target_matrix = phase_to_superpixel(return_array, self.x)
        return target_matrix[:, :, np.newaxis].astype(np.uint8)

    def alignment_pattern_horizontal(self):
        a = np.ones((912, 4)) * 255
        b = np.zeros((912, 4))
        return_array = np.zeros((912, 0))
        for i in range(143):
            return_array = np.hstack((return_array, a))
            return_array = np.hstack((return_array, b))
        return return_array[:, :, np.newaxis].astype(np.uint8)

    def alignment_pattern_vertical(self):
        a = np.ones((4, 1144)) * 255
        b = np.zeros((4, 1144))
        return_array = np.zeros((0, 1144))
        for i in range(114):
            return_array = np.vstack((return_array, a))
            return_array = np.vstack((return_array, b))
        return return_array[:, :, np.newaxis].astype(np.uint8)
    def LG_mode(self,l ,p, range=(-3,3), plot=True):
        min,max = range
        x = np.linspace(min, max, 285)
        y = np.linspace(min, max, 228)
        X, Y = np.meshgrid(x, y)
        Z = gaussian_laguerre_cartesian(l, p, X, Y)
        Z = Z / np.max(abs(Z))
        target_matrix = phase_to_superpixel(Z, self.x)
        Z = abs(Z)
        if plot:
            plot_gaussian_laguerre_cartesian(l,p, X, Y, Z)
            plot_pixel(target_matrix)
        return target_matrix[:, :, np.newaxis].astype(np.uint8)
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
    rho_squared = x ** 2 + y ** 2
    laguerre_term = genlaguerre(p, l)(2*rho_squared)
    angular_term = np.exp(1j * l * np.arctan2(y, x))
    #Normalization

    return np.exp(-rho_squared) * laguerre_term * angular_term

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
def alignment_process():
    SP = SuperpixelPattern(4)
    dmd_pattern_one = SP.alignment_pattern_horizontal()
    dmd_pattern_two = SP.alignment_pattern_vertical()

    dmd = DMD_driver()
    # Create a default project
    dmd.create_project(project_name='test_project')
    dmd.create_main_sequence(seq_rep_count=1)
    # Image
    count = 0
    reference_pattern = SP.plane_wave(0)
    for i in range(0, 200):
        count += 1
        dmd.add_sequence_item(image=dmd_pattern_one, seq_id=1, frame_time=2000)
        dmd.add_sequence_item(image=dmd_pattern_two, seq_id=1, frame_time=2000)
        dmd.add_sequence_item(image=reference_pattern, seq_id=1, frame_time=2000)

    # Create the main sequence
    print("start projecting")
    dmd.my_trigger()
    dmd.start_projecting()
    # Stop the sequence
    time.sleep(500)
    dmd.stop_projecting()

def target_field():
    SP = SuperpixelPattern(4)

    dmd_pattern = SP.LG_mode(1,0)
    dmd = DMD_driver()
    # Create a default project
    dmd.create_project(project_name='test_project')
    dmd.create_main_sequence(seq_rep_count=1)
    # Image
    reference_pattern = np.zeros((912, 1140))[:, :, np.newaxis].astype(np.uint8)
    count = 0
    for i in range(0, 500):
        count += 1
        dmd.add_sequence_item(image=dmd_pattern, seq_id=1, frame_time=3000)
        dmd.add_sequence_item(image=reference_pattern, seq_id=1, frame_time=3000)

    # Create the main sequence
    print("start projecting")
    dmd.my_trigger()
    dmd.start_projecting()
    # Stop the sequence
    time.sleep(500)
    dmd.stop_projecting()
SP = SuperpixelPattern(4)
plot_pixel(SP.LG_mode(2,1))

#alignment_process()
#target_field()