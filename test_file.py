import numpy as np
from itertools import combinations
import math
from decimal import Decimal, getcontext
import matplotlib.pyplot as plt
y = np.e**(np.arange(0,9,1)*2*np.pi/9*1j)/9
x = np.e**(np.arange(0,16,1)*2*np.pi/16*1j)/16
a = np.array([1,2,3])

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
sum_array, combo = get_combination_sums_and_indices(x)
sum_array, combo = remove_complex_duplicates(sum_array, combo)

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
