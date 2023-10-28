import numpy as np
from generate_pattern import DmdPattern
from simulation_DMD import SimulationCompressingImage, plot_pixel
from test_wavelet import *

def embed_in_corner(smaller_matrices, size, position):
    # Convert the smaller_matrices to a list for easy indexing
    smaller_matrices = list(smaller_matrices)

    # Determine the starting row and column based on the position of the first matrix
    first_matrix_shape = smaller_matrices[0].shape
    if position == 1:  # top-left
        start_row, start_col = 0, 0
    elif position == 2:  # top-right
        start_row, start_col = 0, size[1] - first_matrix_shape[1]
    elif position == 3:  # bottom-left
        start_row, start_col = size[0] - first_matrix_shape[0], 0
    elif position == 4:  # bottom-right
        start_row, start_col = size[0] - first_matrix_shape[0], size[1] - first_matrix_shape[1]
    else:
        raise ValueError("Position must be between 1 and 4 inclusive.")

    # Embed each smaller matrix into a new larger matrix
    def embed(matrix):
        matrix = matrix * 256
        larger = np.ones(size) * 128
        larger[start_row:start_row + matrix.shape[0], start_col:start_col + matrix.shape[1]] = matrix
        return larger

    return list(map(embed, smaller_matrices))
def conjugate_pattern(pattern):
        pattern = (pattern==0).astype(int)
        return pattern
def analyse_boarder():
    include_boarde
pattern = DmdPattern("hadamard", 32, 32)
pattern = pattern.execute(two_dimension=True)
reverse_pattern = map(conjugate_pattern, pattern)
pattern = embed_in_corner(pattern, (64,64), 2)
reverse_pattern = embed_in_corner(reverse_pattern, (64,64),2)


SCI = SimulationCompressingImage(64,64)
image = SCI.execute(pattern, reverse_pattern)
plot_pixel(image)
print(image)