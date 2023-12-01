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
def split_matrix(arr):
    width = arr.shape[0]
    top_left, top_right, bot_left, bot_right = (arr[0:int(width/2), 0:int(width/2)],
                                                arr[0:int(width/2), int(width/2):width],
                                                arr[int(width/2):width, 0:int(width/2)],
                                                arr[int(width/2):width,int(width/2):width])
    return  top_left, top_right, bot_left, bot_right



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
def adaptive_pattern(image, size, num_stage):
    image_size = image.shape
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
    projection_list = [1]
    while len(new_pattern_list)>0:
        print(len(new_pattern_list))
        temp_pattern = new_pattern_list[0]
        temp_inverse = negative_pattern_list[0]
        del new_pattern_list[0]
        del negative_pattern_list[0]
        if projection_list[i]==1:
            image_ = SimulationCompressingImage(image).execute(temp_pattern, temp_inverse)
            wt_level_ = wavelet_transform_level_one(image_)
            temp_wt_list.append(wt_level_)

        elif projection_list[i]==0:
            image_ = np.zeros((image_size))
            wt_level_ = wavelet_transform_level_one(image_)
            temp_wt_list.append(wt_level_)
        i+=1
        if i == 4**j:
            print(i, 4**j)
            j+=1
            i=0
            temp_wt_dic = {f"{j}": temp_wt_list}
            wt_dic.update(temp_wt_dic)
            projection_list = []
            for m in range(len(temp_wt_list)):
                print(analyze_wavelet(temp_wt_list[m]))
                projection_list += analyze_wavelet(temp_wt_list[m])
            temp_wt_list=[]
    return wt_dic
def analyze_wavelet(wavelet, threshold=1):
    _, hor, ver, dia = wavelet
    combine_wavelet = hor+ver+dia
    sliced_matrix = split_matrix(combine_wavelet)
    projection_index = []
    for i in range(len(sliced_matrix)):
        summed_value = np.sum(sliced_matrix[i])
        print(summed_value)
        if np.abs(summed_value)<threshold:
            projection_index.append(0)
        else:
            projection_index.append(1)
    return projection_index
def reconstruct_image_from_wavelet(wt_dic):
    reconstruction = []
    for i in range(len(wt_dic)):
        print("len", len(wt_dic))
        temp_wt = wt_dic[f"{i+1}"]
        temp_reconstruction = []
        for j in range(len(temp_wt)):
            print("j", j)
            image, hor, ver, dia = temp_wt[j]
            summed_edge = hor+ver+dia
            print("calculating_edge:", np.sum(summed_edge))
            reconstructed_image = inverse_wavelet_trasform(image, hor, ver, dia)
            temp_reconstruction.append(reconstructed_image)
        reconstruction.append(np.sum(np.array(temp_reconstruction),axis = 0))
    return reconstruction
random_image = random_image_with_shapes(128,128)
plot_pixel(random_image)
wt_list = adaptive_pattern(random_image,32, 2)
image, hor, ver, dia  = wt_list["2"][2]

reconstruction = reconstruct_image_from_wavelet(wt_list)
plot_pixel(reconstruction[0])
plot_pixel(np.sum(np.array(reconstruction), axis=0))