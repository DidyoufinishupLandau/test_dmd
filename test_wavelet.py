import pywt
import numpy as np
from generate_pattern import DmdPattern
from simulation_DMD import SimulationCompressingImage, random_image_with_shapes, plot_pixel
def wavelet_transform_level_one(matrix):
    # Apply single level Discrete Wavelet Transform
    coeffs = pywt.dwt2(matrix, 'db1')  # 'db1' refers to the Daubechies wavelet with one vanishing moment.
    cA, (cH, cV, cD) = coeffs  # cA: approximation, cH: horizontal details, cV: vertical details, cD: diagonal details

    return cA, cH, cV, cD
def inverse_wavelet_trasform(cA, cH, cV, cD):
    reconstructed_data_2D = pywt.idwt2((cA, (cH, cV, cD)), 'db1')
    return reconstructed_data_2D
def embed_in_corner(smaller_matrices, size, position):

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
        if (matrix<2).all():
            matrix = matrix * 256
        larger = np.ones(size) * 128
        larger[start_row:start_row + matrix.shape[0], start_col:start_col + matrix.shape[1]] = matrix
        return larger

    return list(map(embed, smaller_matrices))
def split_image(arr):
    width = arr.shape[0]
    top_left, top_right, bot_left, bot_right = (arr[0:int(width/2), 0:int(width/2)],
                                                arr[0:int(width/2), int(width/2):width],
                                                arr[int(width/2):width, 0:int(width/2)],
                                                arr[int(width/2):width,int(width/2):width])
    return  top_left, top_right, bot_left, bot_right

def test_dimulation_DMD():
    random_image = random_image_with_shapes(128, 128, 10)
    pattern, inverse = DmdPattern("hadamard", 64,64).execute()
    image = SimulationCompressingImage(random_image).execute(pattern, inverse)
    plot_pixel(random_image)
    plot_pixel(image)


def adaptive_pattern(image, size, num_stage):
    gp = DmdPattern('hadamard', size, size)
    two_dimension_hadamard, negative_pattern = gp.execute()
    new_pattern_list = []
    new_pattern_list.append(two_dimension_hadamard)
    negative_pattern_list = []
    negative_pattern_list.append(negative_pattern)
    # generate all pattern that would require in later procedure
    # noting that the order is always from top left to bottom right
    for i in range(1, num_stage+1):
        for j in range(1,5):
            for k in range(4 ** (i - 1)):
                print(int(k + 4 ** (i - 2)), "size", (size*2**(i), size*2**(i)), j)
                negative_pattern_list.append(
                    embed_in_corner(negative_pattern_list[int(k + 4 ** (i - 2))],(size*2**(i), size*2** (i)), j))
                new_pattern_list.append(
                    embed_in_corner(new_pattern_list[int(k + 4 ** (i - 2))], (size*2**(i), size*2**(i)), j))
    wt_dic = {}
    i = 0
    j = 0
    temp_wt_list = []
    while len(new_pattern_list)>0:
        print(len(new_pattern_list))
        temp_pattern = new_pattern_list[0]
        temp_inverse = negative_pattern_list[0]
        del new_pattern_list[0]
        del negative_pattern_list[0]
        i+=1
        image_ = SimulationCompressingImage(image).execute(temp_pattern, temp_inverse)
        wt_level_ = wavelet_transform_level_one(image_)
        temp_wt_list.append(wt_level_)
        if i == 4**j:
            print(i, 4**j)
            j+=1
            i=0
            temp_wt_list = {f"{j}": temp_wt_list}
            wt_dic.update(temp_wt_list)
            temp_wt_list=[]
    return wt_dic
def test_wavelet():
    random_image = random_image_with_shapes(128, 128, 10)
    pattern, inverse = DmdPattern("hadamard", 64,64).execute()
    plot_pixel(random_image)
    # calculate wavelet transform
    image = SimulationCompressingImage(random_image).execute(pattern, inverse)
    wt_level = wavelet_transform_level_one(image) # 64

    plot_pixel(image)
    #now embed in corner
    pattern_one = embed_in_corner(pattern, (128,128), 1)
    inverse_one = embed_in_corner(inverse, (128,128), 1)
    image_one = SimulationCompressingImage(random_image).execute(pattern_one, inverse_one)
    wt_level_one = wavelet_transform_level_one(image_one) # 64
    pattern_two = embed_in_corner(pattern, (128,128), 2)
    inverse_two = embed_in_corner(inverse, (128,128), 2)
    image_two = SimulationCompressingImage(random_image).execute(pattern_two, inverse_two)
    wt_level_two = wavelet_transform_level_one(image_two)
    pattern_three = embed_in_corner(pattern, (128,128), 3)
    inverse_three = embed_in_corner(inverse, (128,128), 3)
    image_three = SimulationCompressingImage(random_image).execute(pattern_three, inverse_three)
    wt_level_three = wavelet_transform_level_one(image_three)
    pattern_three = embed_in_corner(pattern, (128,128), 4)
    inverse_three = embed_in_corner(inverse, (128,128), 4)
    image_four = SimulationCompressingImage(random_image).execute(pattern_three, inverse_three)
    wt_level_four = wavelet_transform_level_one(image_four)
    plot_pixel(image_one+image_two)
    def combine_wavelet(list_wavelet):
        image = []
        horizontal = []
        vertial = []
        diagonal = []
        for i in range(len(list_wavelet)):
            image.append(list_wavelet[i][0])
            horizontal.append(list_wavelet[i][1])
            vertial.append(list_wavelet[i][2])
            diagonal.append(list_wavelet[i][3])
        image_arr = np.array(image)
        horizontal_arr = np.array(horizontal)
        vertial_arr = np.array(vertial)
        diagonal_arr = np.array(diagonal)
        return np.sum(image_arr,axis=0), np.sum(horizontal_arr,axis=0), np.sum(vertial_arr,axis=0),np.sum(diagonal_arr,axis=0)
    list_wavelet = [wt_level_one, wt_level_four]
    ca, ch,cv,cd = combine_wavelet(list_wavelet)
    reconstruction = inverse_wavelet_trasform(ca, ch,cv,cd)
    plot_pixel(reconstruction)
def reconstruct_image_from_wavelet(wt_dic):
    reconstruction = []
    for i in range(len(wt_dic)):
        print("len", len(wt_dic))
        temp_wt = wt_dic[f"{i+1}"]
        temp_reconstruction = []
        for j in range(len(temp_wt)):
            print("j", j)
            image, hor, ver, dia = temp_wt[j]
            reconstructed_image = inverse_wavelet_trasform(image, hor, ver, dia)
            temp_reconstruction.append(reconstructed_image)
        reconstruction.append(np.sum(np.array(temp_reconstruction),axis = 0))
    return reconstruction
random_image = random_image_with_shapes(128,128)
plot_pixel(random_image)
wt_list = adaptive_pattern(random_image,32, 2)

reconstruction = reconstruct_image_from_wavelet(wt_list)
plot_pixel(reconstruction[0])
plot_pixel(reconstruction[1])
plot_pixel(reconstruction[2])