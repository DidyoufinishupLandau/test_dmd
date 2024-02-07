import numpy as np
from itertools import combinations
from decimal import Decimal, getcontext
import matplotlib.pyplot as plt
from scipy.special import genlaguerre
from simulation_DMD import plot_pixel
from DMD_driver import DMD_driver
import time
from scipy.special import hermite, factorial




y = np.e**(np.arange(0,9,1)*2*np.pi/9*1j)
y = np.e**(np.arange(0,9,1)*2*np.pi/9*1j)/9
x = np.e**(np.arange(0,16,1)*2*np.pi/16*1j)/16
a = np.array([1,2,3])
def hermite_gaussian(x, y, z, w0, k, m, n):
    """
    Generate the intensity distribution of a Hermite-Gaussian beam.

    Parameters:
        x (numpy.ndarray): x-coordinate.
        y (numpy.ndarray): y-coordinate.
        z (float): Axial distance.
        w0 (float): Beam waist.
        k (float): Wavenumber.
        m (int): Mode index in x-direction.
        n (int): Mode index in y-direction.

    Returns:
        numpy.ndarray: Intensity distribution.
    """
    # Calculate beam radius at distance z
    x, y = np.meshgrid(x, y)
    w_z = w0 * np.sqrt(1 + (z * k * w0**2 / 2)**2)

    # Normalization factor
    C_nm = np.sqrt((2**(-1 - m - n) * factorial(m) * factorial(n)) / (np.pi * factorial(m + n)))

    # Hermite polynomials
    H_m = hermite(m)(np.sqrt(2) * x / w_z)
    H_n = hermite(n)(np.sqrt(2) * y / w_z)

    # Gaussian envelope
    gaussian = np.exp(-(x**2 + y**2) / w_z**2)

    # Intensity distribution
    intensity = C_nm * H_m * H_n * gaussian

    return np.abs(intensity)**2/np.max(np.abs(intensity))**2, intensity/np.max(np.abs(intensity))

def plot_intensity(x, y, intensity):
    """
    Plot the intensity distribution.

    Parameters:
        x (numpy.ndarray): x-coordinate values.
        y (numpy.ndarray): y-coordinate values.
        intensity (numpy.ndarray): Intensity values.
    """
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(x, y, intensity, shading='auto')
    plt.colorbar(label='Intensity')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Intensity Distribution of Hermite-Gaussian Beam')
    plt.grid(True)
    plt.show()
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


def plot_complex_array(complex_array, figure_size=(20, 20)):
    real_part = [num.real for num in complex_array]
    imag_part = [num.imag for num in complex_array]
    plt.figure(figsize=figure_size)
    plt.scatter(real_part, imag_part, marker='x', color='blue')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('Complex Numbers in Complex Plane')
    plt.grid(True)
    plt.show()


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
    laguerre_term = genlaguerre(p, l)(2*rho_squared)

    # Calculate the angular term
    angular_term = np.exp(1j * l * np.arctan2(y, x))

    # Normalize the Laguerre-Gaussian beam
    normalized_beam = np.exp(-rho_squared)*laguerre_term * angular_term

    return normalized_beam
def plane_wave():
    return_array = np.zeros((228,286))
    return return_array
def alignment_pattern_horizontal():
    a = np.ones((912,4))*255
    b = np.zeros((912,4))
    return_array = np.zeros((912,0))
    for i in range(143):
        return_array = np.hstack((return_array,a))
        return_array = np.hstack((return_array,b))
    return return_array[:,:, np.newaxis].astype(np.uint8)
def alignment_pattern_vertical():
    a = np.ones((4,1144))*255
    b = np.zeros((4,1144))
    return_array = np.zeros((0,1144))
    for i in range(114):
        return_array = np.vstack((return_array,a))
        return_array = np.vstack((return_array,b))
    return return_array[:,:, np.newaxis].astype(np.uint8)


def plot_gaussian_laguerre_cartesian(l, p):
    x = np.linspace(-3,3, 285)
    y = np.linspace(-3, 3, 228)
    X, Y = np.meshgrid(x, y)
    Z = gaussian_laguerre_cartesian(l, p, X, Y)
    normalized_E_field = Z/np.max(abs(Z))
    Z = np.abs(normalized_E_field)

    plt.contourf(X, Y, Z, cmap='viridis', levels=100)
    plt.colorbar(label='Intensity')
    plt.title(f'Gaussian Laguerre, l={l}, p={p}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    return normalized_E_field
def phase_to_superpixel(phase_matrix, error=10**-2):
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
    sum_array, combo = get_combination_sums_and_indices(x)
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

def alignment_process():
    dmd_pattern_one = alignment_pattern_horizontal()
    dmd_pattern_two = alignment_pattern_vertical()

    dmd = DMD_driver()
    # Create a default project
    dmd.create_project(project_name='test_project')
    dmd.create_main_sequence(seq_rep_count=1)
    # Image
    count = 0
    sum_array, combo = get_combination_sums_and_indices(x)
    sum_array, combo = remove_complex_duplicates(sum_array, combo)
    normalized_E_field = plot_gaussian_laguerre_cartesian(0, 2)
    # normalized_E_field = plane_wave()
    target_matrix = phase_to_superpixel(normalized_E_field)
    reference_pattern = target_matrix[:, :, np.newaxis].astype(np.uint8)
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
    sum_array, combo = get_combination_sums_and_indices(x)
    sum_array, combo = remove_complex_duplicates(sum_array, combo)
    normalized_E_field = plot_gaussian_laguerre_cartesian(0, 2)
    # normalized_E_field = plane_wave()
    target_matrix = phase_to_superpixel(normalized_E_field)
    dmd_pattern = target_matrix[:, :, np.newaxis].astype(np.uint8)
    print(dmd_pattern)
    dmd = DMD_driver()
    # Create a default project
    dmd.create_project(project_name='test_project')
    dmd.create_main_sequence(seq_rep_count=1)
    # Image
    reference_pattern = np.ones((228, 285))
    reference_pattern = phase_to_superpixel(reference_pattern)[:, :, np.newaxis].astype(np.uint8)
    plot_pixel(reference_pattern)
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

def project_gaussian_hermite_beam():
    w0 = 1.0  # Beam waist
    k = 2 * np.pi / 0.5  # Wavenumber
    gh_target_matrix_list = []
    for i in range(2):
        for j in range(2):
            m = i  # Mode index in x-direction
            n = j  # Mode index in y-direction
            # Calculate intensity distribution
            gaussian_hermite, gaussian_hermite_phase = hermite_gaussian(np.linspace(-5, 5, 285),
                                                                        np.linspace(-5, 5, 228),
                                                                        0, w0, k, m, n)
            gh_target_matrix = phase_to_superpixel(gaussian_hermite_phase)
            gh_target_matrix_list.append(gh_target_matrix[:, :, np.newaxis].astype(np.uint8))

    # Plot intensity distribution
    plot_intensity(np.linspace(-5, 5, 285), np.linspace(-5, 5, 228), gaussian_hermite)
    plot_pixel(gh_target_matrix)

    dmd = DMD_driver()
    # Create a default project
    dmd.create_project(project_name='test_project')
    dmd.create_main_sequence(seq_rep_count=1)
    # Image
    reference_pattern = np.ones((228, 285))
    reference_pattern = phase_to_superpixel(reference_pattern)[:, :, np.newaxis].astype(np.uint8)
    plot_pixel(reference_pattern)
    count = 0
    for i in range(0, 50):
        count += 1
        for j in range(len(gh_target_matrix_list)):
            dmd.add_sequence_item(image=gh_target_matrix_list[j], seq_id=1, frame_time=3000)
        dmd.add_sequence_item(image=reference_pattern, seq_id=1, frame_time=3000)
    # Create the main sequence
    print("start projecting")
    dmd.my_trigger()
    dmd.start_projecting()
    # Stop the sequence
    time.sleep(500)
    dmd.stop_projecting()
#alignment_process()
#target_field()
project_gaussian_hermite_beam()
