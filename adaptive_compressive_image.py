from generate_pattern import DmdPattern
import numpy as np
from simulation_DMD import SimulationCompressingImage
from DMD_driver import DMD_driver
import pywt
import cv2
from stream_arduino import StreamArduino, save_data
import sys
from simulation_DMD import plot_pixel
class SimulateAdaptiveCompressiveImage:
    def __init__(self, num_stage, image_resolution, threshold):
        """
        The class for implementing adaptive compressive image algorithm
        :param num_stage: number of stage an integer
        :param image_resolution: The resolution of final image
        :param threshold: Any wavelet with summed up value below threshold will be abondoned.
        """
        self.num_stage = num_stage
        self.Hadamard_size = int(image_resolution/2**(num_stage))
        self.threshold = threshold

    def adaptive_pattern(self, image):
        image_size = image.shape
        #print(self.Hadamard_size)
        gp = DmdPattern('hadamard', self.Hadamard_size, self.Hadamard_size)
        two_dimension_hadamard, negative_pattern = gp.execute()
        new_pattern_list = []
        new_pattern_list.append(two_dimension_hadamard)
        negative_pattern_list = []
        negative_pattern_list.append(negative_pattern)
        # generate all pattern that would require in later procedure
        # noting that the order is always from top left to bottom right
        for i in range(1, self.num_stage + 1):
            for j in range(1, 5):
                for k in range(4 ** (i - 1)):
                    #print(int(k + 4 ** (i - 2)), "size", (self.Hadamard_size * 2 ** (i), self.Hadamard_size * 2 ** (i)), j)
                    negative_pattern_list.append(
                        embed_in_corner(negative_pattern_list[int(k + 4 ** (i - 2))],
                                        (self.Hadamard_size * 2 ** (i), self.Hadamard_size * 2 ** (i)), j))
                    new_pattern_list.append(
                        embed_in_corner(new_pattern_list[int(k + 4 ** (i - 2))], (self.Hadamard_size * 2 ** (i), self.Hadamard_size * 2 ** (i)), j))
        wt_dic = {}
        i = 0
        j = 0
        temp_wt_list = []
        projection_list = [1]
        while len(new_pattern_list) > 0:
            #print("new_patternlist", len(new_pattern_list))
            temp_pattern = new_pattern_list[0]
            temp_inverse = negative_pattern_list[0]
            del new_pattern_list[0]
            del negative_pattern_list[0]
            if projection_list[i] == 1:
                #replace the following line to DMD main
                image_ = SimulationCompressingImage(image).execute(temp_pattern, temp_inverse)
                wt_level_ = wavelet_transform_level_one(image_)
                temp_wt_list.append(wt_level_)

            elif projection_list[i] == 0:

                image_ = np.zeros((image_size))
                wt_level_ = wavelet_transform_level_one(image_)
                temp_wt_list.append(wt_level_)
            i += 1
            if i == 4 ** j:
                #print(i, 4 ** j)
                j += 1
                i = 0
                temp_wt_dic = {f"{j}": temp_wt_list}
                wt_dic.update(temp_wt_dic)
                projection_list = []
                for m in range(len(temp_wt_list)):
                    #print("analyzing",analyze_wavelet(temp_wt_list[m], threshold=self.threshold))
                    projection_list += analyze_wavelet(temp_wt_list[m], threshold=self.threshold)
                temp_wt_list = []
        return wt_dic

    def reconstruct_image_from_wavelet(self, wt_dic):
        reconstruction = []
        for i in range(len(wt_dic)):
            #print("len", len(wt_dic))
            temp_wt = wt_dic[f"{i + 1}"]
            temp_reconstruction = []
            for j in range(len(temp_wt)):
                #print("j", j)
                image, hor, ver, dia = temp_wt[j]
                #print("calculating_edge:", np.sum(summed_edge))
                reconstructed_image = inverse_wavelet_trasform(image, hor, ver, dia)
                temp_reconstruction.append(reconstructed_image)
            reconstruction.append(np.sum(np.array(temp_reconstruction), axis=0))

        return reconstruction
class AdaptiveCompressiveImage:
    def __init__(self, num_stage, image_resolution,group, threshold=10, frame_time=1):
        """
        The class for implementing adaptive compressive image algorithm
        :param num_stage: number of stage an integer
        :param image_resolution: The resolution of final image
        :param threshold: Any wavelet with summed up value below threshold will be abondoned.
        """
        self.num_stage = num_stage
        self.Hadamard_size = int(image_resolution/2**(num_stage))
        self.threshold = threshold
        self.frame_time = frame_time
        self.group = group


    def adaptive_pattern(self):
        #print(self.Hadamard_size)
        gp = DmdPattern('hadamard', self.Hadamard_size, self.Hadamard_size)
        two_dimension_hadamard, negative_pattern = gp.execute()
        new_pattern_list = []
        new_pattern_list.append(two_dimension_hadamard)
        negative_pattern_list = []
        negative_pattern_list.append(negative_pattern)
        # generate all pattern that would require in later procedure
        # noting that the order is always from top left to bottom right
        for i in range(1, self.num_stage + 1):
            for j in range(1, 5):
                for k in range(4 ** (i - 1)):
                    #print(int(k + 4 ** (i - 2)), "size", (self.Hadamard_size * 2 ** (i), self.Hadamard_size * 2 ** (i)), j)
                    negative_pattern_list.append(
                        embed_in_corner(negative_pattern_list[int(k + 4 ** (i - 2))],
                                        (self.Hadamard_size * 2 ** (i), self.Hadamard_size * 2 ** (i)), j))
                    new_pattern_list.append(
                        embed_in_corner(new_pattern_list[int(k + 4 ** (i - 2))], (self.Hadamard_size * 2 ** (i), self.Hadamard_size * 2 ** (i)), j))
        return  new_pattern_list, negative_pattern_list
    def execute(self, image_size, measurement_size=4096):
        new_pattern_list, negative_pattern_list = self.adaptive_pattern()
        dmd_pattern_list = []
        switch = True
        #initialize dmd
        while switch:
            self.group += 1
            wt_dic = {}
            counter = 0
            j = 0
            temp_wt_list = []
            projection_list = [1]
            for i in range(len(new_pattern_list)):
                if  projection_list[counter] == 1:
                    dmd_pattern = list(map(rescale_cv, new_pattern_list[i]))
                    # print("new_patternlist", len(new_pattern_list))
                    #replace the following line to DMD main
                    image_ = adaptive_dmd(0,
                                          len(dmd_pattern),
"projectname",
                                          1,
                        dmd_pattern,
                                          self.frame_time,
                                          self.group,
                        i,
                                          new_pattern_list[i],
                                          negative_pattern_list[i]
                                          )
                    plot_pixel(image_)
                    wt_level_ = wavelet_transform_level_one(image_)
                    temp_wt_list.append(wt_level_)

                elif projection_list[counter] == 0:

                    image_ = np.zeros((image_size))
                    wt_level_ = wavelet_transform_level_one(image_)
                    temp_wt_list.append(wt_level_)
                counter += 1
                if counter == 4 ** j:
                    print(counter, 4 ** j)
                    j += 1
                    counter = 0
                    temp_wt_dic = {f"{j}": temp_wt_list}
                    wt_dic.update(temp_wt_dic)
                    projection_list = []
                    for m in range(len(temp_wt_list)):
                        print("analyzing",analyze_wavelet(temp_wt_list[m], threshold=self.threshold))
                        projection_list += analyze_wavelet(temp_wt_list[m], threshold=self.threshold)
                    temp_wt_list = []
            print("end_process type:")
            line = sys.stdin.readline()
            if line == 'TRUE\n':
                switch = True
            elif line == 'STOP\n':
                switch = False
        return wt_dic

    def reconstruct_image_from_wavelet(self, wt_dic):
        reconstruction = []
        for i in range(len(wt_dic)):
            #print("len", len(wt_dic))
            temp_wt = wt_dic[f"{i + 1}"]
            temp_reconstruction = []
            for j in range(len(temp_wt)):
                #print("j", j)
                image, hor, ver, dia = temp_wt[j]
                summed_edge = hor + ver + dia
                #print("calculating_edge:", np.sum(summed_edge))
                reconstructed_image = inverse_wavelet_trasform(image, hor, ver, dia)
                temp_reconstruction.append(reconstructed_image)
            reconstruction.append(np.sum(np.array(temp_reconstruction), axis=0))
        return reconstruction
    def reconstruct_image_from_wavelet(self, wt_dic):
        reconstruction = []
        for i in range(len(wt_dic)):
            #print("len", len(wt_dic))
            temp_wt = wt_dic[f"{i + 1}"]
            temp_reconstruction = []
            for j in range(len(temp_wt)):
                #print("j", j)
                image, hor, ver, dia = temp_wt[j]
                #print("calculating_edge:", np.sum(summed_edge))
                reconstructed_image = inverse_wavelet_trasform(image, hor, ver, dia)
                temp_reconstruction.append(reconstructed_image)
            reconstruction.append(np.sum(np.array(temp_reconstruction), axis=0))
        image = reconstruction[0]
        plot_pixel(image)
        for i in range(1, len(reconstruction)):
            _, hor, ver, dia = wavelet_transform_level_one(reconstructed_image[i])
            image = inverse_wavelet_trasform(image, hor, ver, dia)
            plot_pixel(image)

        return reconstruction
def adaptive_dmd(start,
                 end,
                 name,
                 iter,
                 pattern,
                 frame_time,
                 group,
                 index,
                 positive_pattern,
                 negative_pattern):
    measurement_size = len(positive_pattern)
    dmd = DMD_driver()
    dmd.create_project(project_name=name)
    dmd.create_main_sequence(seq_rep_count=iter)
    for j in range(start, end):
        dmd.add_sequence_item(image=pattern[j], seq_id=1, frame_time=frame_time)
    dmd.my_trigger()
    ############################
    ser = StreamArduino()
    ser.start()
    ###########################
    print("sleep:", frame_time / 1000 * 1.01 * (end - start) * iter)
    dmd.start_projecting()
    ############################ get data and stop projecting
    data_one, data_two = ser.get_data_for_adaptive(measurement_size, iter)
    dmd.stop_projecting()
    ser.close()
    intensity = data_two-data_one
    ########### save data
    image = []
    pattern = np.array(positive_pattern) - np.array(negative_pattern)
    length_pattern = len(positive_pattern)
    intensity_length = len(intensity)
    for i in range(len(intensity)):
        image.append((intensity[i] * pattern[i]).astype(np.float64))
    image = np.sum(np.array(image), axis=0) / length_pattern
    return image
def rescale_cv(input_image):
    """
    resize any size image into 1140 * 912 image.
    :param input_image: single mask
    :return:
    2D array: single mask
    """
    return cv2.resize(input_image*255, (912, 1140), interpolation=cv2.INTER_LINEAR).astype(np.uint8)[:,:, np.newaxis]
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
        if (matrix<2).all():
            matrix = matrix * 256
        larger = np.ones(size) * 128
        larger[start_row:start_row + matrix.shape[0], start_col:start_col + matrix.shape[1]] = matrix
        return larger

    return list(map(embed, smaller_matrices))
def reshape_image(two_dimension_image):
    two_dimension_image = two_dimension_image
    return two_dimension_image.T[:, :, np.newaxis]
def analyze_wavelet(wavelet, threshold=1):
    _, hor, ver, dia = wavelet
    combine_wavelet = hor+ver+dia
    sliced_matrix = split_matrix(combine_wavelet)
    projection_index = []
    for i in range(len(sliced_matrix)):
        summed_value = np.sum(sliced_matrix[i])
        #print("waveletinfo", np.abs(summed_value))
        if np.abs(summed_value)<threshold:
            projection_index.append(0)
        else:
            projection_index.append(1)
    return projection_index
def wavelet_transform_level_one(matrix):
    # Apply single level Discrete Wavelet Transform
    coeffs = pywt.dwt2(matrix, 'db1')  # 'db1' refers to the Daubechies wavelet with one vanishing moment.
    cA, (cH, cV, cD) = coeffs  # cA: approximation, cH: horizontal details, cV: vertical details, cD: diagonal details

    return cA, cH, cV, cD
def inverse_wavelet_trasform(cA, cH, cV, cD):
    reconstructed_data_2D = pywt.idwt2((cA, (cH, cV, cD)), 'db1')
    return reconstructed_data_2D

def split_matrix(arr):
    width = arr.shape[0]
    top_left, top_right, bot_left, bot_right = (arr[0:int(width/2), 0:int(width/2)],
                                                arr[0:int(width/2), int(width/2):width],
                                                arr[int(width/2):width, 0:int(width/2)],
                                                arr[int(width/2):width,int(width/2):width])
    return  top_left, top_right, bot_left, bot_right
