import generate_pattern
from stream_raspberry import StreamRaspberry
from test_wavelet import *
from read_and_reconstruct import combine_data
from DMD_main import control_DMD
from skimage import transform

class AdaptiveCompressiveImage:
    def __init__(self, num_stage, image_resolution, threshold):
        """
        The class for implementing adaptive compressive image algorithm
        :param num_stage: number of stage an integer
        :param image_resolution: The resolution of final image
        :param threshold: Any wavelet with summed up value below threshold will be abondoned.
        """
        self.num_stage = num_stage
        self.image_resolution = image_resolution
        self.Hadamard_size = int(image_resolution/2**(num_stage-1))
        self.threshold = threshold
    def adaptive_pattern(self):
        gp = generate_pattern.DmdPattern('hadamard', self.Hadamard_size, self.Hadamard_size)
        two_dimension_hadamard, negative_pattern = gp.execute(two_dimension=True)
        new_pattern_list = []
        new_pattern_list.append(two_dimension_hadamard)
        negative_pattern_list = []
        negative_pattern_list.append(negative_pattern)
        # generate all pattern that would require in later procedure
        # noting that the order is always from top left to bottom right
        for i in range(1, self.num_stage):
            for j in range(4):
                for k in range(4 ** (i - 1)):
                    print(int(k+4**(i-2)))
                    negative_pattern_list.append(embed_in_corner(negative_pattern_list[int(k+4**(i-2))], self.Hadamard_size*(i+1), j))
                    new_pattern_list.append(embed_in_corner(new_pattern_list[int(k+4**(i-2))], self.Hadamard_size*(i+1), j))
        return new_pattern_list, negative_pattern_list

    def execute(self):
        # The relevance of quadrant. 1 imply there will be further sampling in this quadrant
        record_quadrant = [1]
        # The wavelet information will save in this list
        wavelet_info = []
        # The positive and negative array[2d array]
        pattern_list, negative_pattern_list = self.adaptive_pattern()
        # The number of frames equal to the number of patterns
        num_frames =pattern_list[0].shape[0]
        for i in range(self.num_stage):
            #Now we define a inner function to rescale the image size
            #The first stage image size is equavalent to image resolution
            # However, the pattern resolution is smaller than image resolution.
            # The second stage image size equal to image resolution / 2
            # The pattern resolution * 2 by taking using same size pattern but focus on smaller region.
            def rescale(input_image: np.array) -> np.array:
                rs = transform.resize(input_image,
                                      (int(self.image_resolution / (2 ** i)), int(self.image_resolution / 2 ** i)),
                                      order=0, anti_aliasing=False)
                return rs

            for j in range(4**i):
                previous_image = len(record_quadrant)
                if record_quadrant[j + previous_image-1] == 1:
                    #initialize data aquisation
                    cd = control_DMD(pattern_list[previous_image+j-1], "adaptive_sampling", 1, 10)
                    SR = StreamRaspberry()
                    data_one, data_two = SR.get_data(num_frames)  # check the correspondence of photodiode and pin
                    cd.execute(0, num_frames)
                    # prepare pattern
                    positive_hadamard = map(rescale, pattern_list[previous_image+j-1])
                    negative_hadamard = map(rescale, negative_pattern_list[previous_image+j-1])
                    #calcualte intensity
                    intensity = (data_one-data_two)/(data_one+data_two)
                    # calculate image
                    image = combine_data(positive_hadamard, negative_hadamard, intensity)
                    # extract image from larger image(remember we only sample a fraction)
                    row, col = map_to_position(j, 4**i)
                    target_image = image[row*self.Hadamard_size:(row+1)*self.Hadamard_size,
                                   col*self.Hadamard_size:(col+1)*self.Hadamard_size]
                    # wavelet transform
                    list_wavelet = wavelet_transform_level_one(target_image)
                    wavelet_info.append(list_wavelet)
                    # devide image into four quadrant and calculate their threshold
                    # if below default threshold, append 0 to 'record_quadrant' list.
                    sliced_matrix = slice_matrix(np.sum(list_wavelet, axis=0), 4)
                    record_quadrant = record_quadrant+list(map(analyze_threshold,sliced_matrix))
                # if there are no relevance, append zero matrix that makes no effect in further action.
                elif record_quadrant[j+previous_image-1] == 0:
                    wavelet_shape = self.image_resolution/2**(i+1)
                    fill_wavelet = np.zeros((wavelet_shape, wavelet_shape))
                    wavelet_info.append((fill_wavelet, fill_wavelet,fill_wavelet,fill_wavelet))

            return wavelet_info
    def inverse_wavelet_transform(self, wavelet_info):
        """
        recombine wavelet transform to reconstruct image.
        :param wavelet_info: a list contain the information of wavelet transform[(coarse image, horizontal, vertial, diagonal)...]
        :return: reconstructed image
        """
        recover_list = []
        #define a function to support combine fractional image to whole picture
        def recombine_data(data_tuple):
            a,b,c,d = data_tuple
            new_list = []
            for i in range(len(a)):
                top = np.hstack((a[i],b[i]))
                bottom = np.hstack((c[i],d[i]))
                recover = np.vstack((top, bottom))
                new_list.append(recover)
            return new_list
        counter = 0
        recover_list.append(wavelet_info[0])
        wavelet_info.pop(0)
        while len(wavelet_info) != 0:
            counter+=1
            temp_wavelet_data = wavelet_info[0:4**counter]
            del wavelet_info[0:4**counter]

            for i in range(counter):
                # now I create an array to inform the correspondant index within wavelet list.
                # The number of fractional wavelet(or image) is 4**k in k stage in total.
                # Further, the order of each fraction is from top left to bottom right.
                # hence, for 4**k stage, there will be at most k repeatation happen.
                # for example if we are recombine stage 2  wavelet information.
                # 4**2 = 16 wavelet fraction in total.
                # we recursively comobine wavelet twice(16->4, 4->1).
                # The remained wavelet is the expected wavelet.

                # We apply this procedure to horizontal, vertial, and diagonal wavelet respectively.
                num_fraction = 4**(counter-i)
                fraction_array = np.linspace(0,num_fraction-1,num_fraction).astype(int)
                fraction_array = fraction_array.reshape(int(len(fraction_array)/4,4))
                for j in range(len(fraction_array)):
                    for k in range(len(fraction_array[j])):
                        temp_list = []
                        temp_list.append(temp_wavelet_data[j][k])
                        temp_wavelet_data.append(recombine_data(temp_list))
                del temp_wavelet_data[0:4**(counter-i)]
            recover_list.append(temp_wavelet_data[0])
        ###############################################inverse wavelet transform
        image = recover_list[-1][0]
        for i in range(len(recover_list)):
            image = inverse_wavelet_trasform(image,
                                             reshape_image[i][1],
                                             reshape_image[i][2],
                                             reshape_image[i][3]
                                             )
        return image

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
def slice_matrix(matrix, num_scene):
    length_matrix = matrix.shape[0]
    matrix_size = int(length_matrix/num_scene)
    scene = []
    for i in range(num_scene):
        for j in range(num_scene):
            scene.append(matrix[matrix_size*i:(matrix_size*i+matrix_size), matrix_size*j:(matrix_size*j+matrix_size)])
    return scene

def analyze_threshold(summed_wavelet, threshold: int= 1000):
    if summed_wavelet>threshold:
        return 1
    else:
        return 0
def map_to_position(n, width):
    row = n // width
    col = n % width
    return row, col
