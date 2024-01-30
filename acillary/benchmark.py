import generate_pattern
from simulation_DMD import random_image_with_shapes, SimulationCompressingImage, plot_pixel, PSNR, resize
from PIL import Image
from adaptive_compressive_image import SimulateAdaptiveCompressiveImage
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
import cv2

def fourier(image):
    dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(image))
    return dark_image_grey_fourier
def inverse_fourier_transform(fourier):
    return abs(np.fft.ifft2(fourier))
def test_adaptive():
    # Load the image
    image_path = 'test_image.png'
    image = Image.open(image_path)

    # Resize the image to 128x128 pixels
    image_resized = image.resize((128, 128))
    # Convert the image to grayscale
    image_gray = image_resized.convert('L')
    image_matrix = np.array(image_gray)
    image_matrix = random_image_with_shapes(128,128)
    image_matrix = rescale_2d_array(image_matrix)
    plot_pixel(image_matrix)
    fourier_transform = np.fft.fftshift(np.fft.fftn(image_matrix))
    plot_pixel(np.log(abs(fourier_transform)))
    # Convert the image to a 2D numpy array
    ################
    #reference image
    pg = generate_pattern.DmdPattern("hadamard", 128,128)
    positive, negative = pg.execute(random_sparsity=0.5, length = 1)
    f = SimulationCompressingImage(image_matrix)
    fourier = f.fourier(positive,negative, 1)
    simulated_image = np.log(abs(fourier))
    ################
    plot_pixel(simulated_image)
    fourier_ground = np.fft.fftshift(np.fft.fft2(image_matrix))
    aci = SimulateAdaptiveCompressiveImage(2, 128, 1)
    pattern = aci.adaptive_pattern(fourier_ground)
    reconstruction = aci.reconstruct_image_from_wavelet(pattern)
    plot_pixel(np.log(abs(reconstruction[0])))
    plot_pixel(np.log(abs(reconstruction[1])))
    plot_pixel(np.log(abs(reconstruction[2])))
    plot_pixel(np.log(abs(np.sum(reconstruction,axis=0))))
    plot_pixel(np.log(abs(np.fft.ifft2(reconstruction[0]))))
    plot_pixel(np.log(abs(np.fft.ifft2(reconstruction[1]))))
    plot_pixel(np.log(abs(np.fft.ifft2(reconstruction[2]))))
    plot_pixel(np.log(abs(np.fft.ifft2(np.sum(reconstruction,axis=0)))))

    pattern = aci.adaptive_pattern(image_matrix)
    reconstruction = aci.reconstruct_image_from_wavelet(pattern)
    plot_pixel(reconstruction[0])
    plot_pixel(reconstruction[1])
    plot_pixel(reconstruction[2])
    plot_pixel(np.sum(reconstruction, axis=0))
    #calculate psnr value
def rescale_2d_array(arr):
    # Normalizing the array to [0, 1]
    min_val = arr.min()
    max_val = arr.max()
    normalized_arr = (arr - min_val) / (max_val - min_val)

    # Rescaling to [-255, 255]
    rescaled_arr = (normalized_arr * 255)

    return rescaled_arr
def test_hadamard():
    image_path = 'test_image.png'
    image = Image.open(image_path)
    image_resized = image.resize((128, 128))
    # Convert the image to grayscale
    image_gray = image_resized.convert('L')
    image_matrix = np.array(image_gray)
    call_pattern = generate_pattern.DmdPattern('hadamard', 128, 128, gray_scale=255)
    hadamard, negative_hadamard= call_pattern.execute(length=1)
    sampling_list = [1,0.5,0.25,0.1]
    PSN = []
    image_list = []
    for i in range(len(sampling_list)):
        SCI = SimulationCompressingImage(image_matrix)
        image = SCI.execute(hadamard, negative_hadamard,sampling_rate=sampling_list[i])
        image = -1*image
        image = rescale_2d_array(image)
        plot_pixel(image)
        image_list.append(image)
        PSN.append(PSNR(image_matrix,image).execute())
    plot_four_images(image_list, PSN)

def plot_four_images(images, psnr_values):
    # Check if exactly four images and four PSNR values are provided
    if len(images) != 4 or len(psnr_values) != 4:
        raise ValueError("Exactly four images and four PSNR values must be provided")

    # Set up the figure size and layout
    fig, axs = plt.subplots(2, 2, figsize=(16, 14))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)  # Adjust spacing to prevent overlapping

    # Flatten the array of axes for easy iteration
    axs = axs.flatten()
    mask_length = [1,0.5,0.25,0.1]
    for i, img in enumerate(images):
        if i ==0:
            im = axs[i].imshow(img, cmap='gray')
            axs[i].axis('off')  # Turn off axis labels

            # Add title with PSNR value
            axs[i].set_title(f"Pattern used {mask_length[i]*100}% - PSNR: GROUND TRUTH ", fontsize=20)

            # Create a colorbar for each image
            plt.colorbar(im, ax=axs[i], orientation='vertical', shrink=0.6)
        # Display each image in its respective subplot
        else:
            im = axs[i].imshow(img, cmap='gray')
            axs[i].axis('off')  # Turn off axis labels

            # Add title with PSNR value
            axs[i].set_title(f"Pattern used {mask_length[i]*100}% - PSNR: {psnr_values[i]:.2f} dB", fontsize=24)

            # Create a colorbar for each image
            plt.colorbar(im, ax=axs[i], orientation='vertical', shrink=0.6)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
def test_random():
    image_path = 'test_image.png'
    image = Image.open(image_path)
    image_resized = image.resize((128, 128))
    # Convert the image to grayscale
    image_gray = image_resized.convert('L')
    image_matrix = np.array(image_gray)
    call_pattern = generate_pattern.DmdPattern('random', 128, 128, gray_scale=255)
    hadamard, negative_hadamard= call_pattern.execute(random_sparsity=0.5, length=2)
    print(len(hadamard))
    sampling_list = [1, 0.5,0.25,0.1]
    PSN = []
    image_list = []
    for i in range(len(sampling_list)):
        SCI = SimulationCompressingImage(image_matrix)
        image = SCI.execute(hadamard, negative_hadamard,sampling_rate=sampling_list[i])
        image = -1*image
        image = rescale_2d_array(image)
        plot_pixel(image)
        image_list.append(image)
        PSN.append(PSNR(image_matrix,image).execute())
    plot_four_images(image_list, PSN)
def test_raster():
    image_path = 'test_image.png'
    image = Image.open(image_path)
    image_resized = image.resize((128, 128))
    # Convert the image to grayscale
    image_gray = image_resized.convert('L')
    image_matrix = np.array(image_gray)
    call_pattern = generate_pattern.DmdPattern('raster', 128, 128, gray_scale=255)
    hadamard, negative_hadamard = call_pattern.execute(length=1)
    sampling_list = [1, 0.5, 0.25, 0.1]
    PSN = []
    image_list = []
    for i in range(len(sampling_list)):
        SCI = SimulationCompressingImage(image_matrix)
        image = SCI.oneside(hadamard, sampling_rate=sampling_list[i])
        image = image
        image = rescale_2d_array(image)
        plot_pixel(image)
        image_list.append(image)
        PSN.append(PSNR(image_matrix, image).execute())
    print(len(image_matrix))
    plot_four_images(image_list, PSN)

def test_fourier():
    image_path = 'test_image.png'
    image = Image.open(image_path)
    image_resized = image.resize((128, 128))
    # Convert the image to grayscale
    image_gray = image_resized.convert('L')
    image_matrix = np.array(image_gray)
    call_pattern = generate_pattern.DmdPattern('fourier', 64, 64, gray_scale=255)
    fourier_pattern= call_pattern.execute(random_sparsity=0.5, length=1)
    image = SimulationCompressingImage(image_matrix).oneside(fourier_pattern)
    plot_pixel(-1*image)

def compare_psnr():
    a = generate_pattern.DmdPattern('hadamard', 128, 128)
    hadamard, negative_hadamard = a.execute()
    b= generate_pattern.DmdPattern('random', 128, 128, gray_scale=255)
    random, random_negative = b.execute(random_sparsity=0.5)
    c= generate_pattern.DmdPattern('raster', 128,128)
    raster, negative = c.execute()
    hadamard_array = np.zeros((0,4))
    random_array = np.zeros((0,4))
    raster_array = np.zeros((0,4))
    for _ in range(10):

        image = random_image_with_shapes(128,128)
        image_matrix = rescale_2d_array(image)
        plot_pixel(image_matrix)
        SCI = SimulationCompressingImage(image_matrix)
        sampling_list = [1, 0.5, 0.25, 0.1]
        temp_ha = []
        temp_ra = []
        temp_ras = []
        for i in range(len(sampling_list)):
            #hadamard
            SCI = SimulationCompressingImage(image_matrix)
            image = SCI.execute(hadamard, negative_hadamard, sampling_rate=sampling_list[i])
            image = -1 * image
            image = rescale_2d_array(image)
            plot_pixel(image)
            temp_ha.append(PSNR(image_matrix, image).execute())

            SCI = SimulationCompressingImage(image_matrix)
            image = SCI.execute(random, random_negative, sampling_rate=sampling_list[i])
            image = -1 * image
            image = rescale_2d_array(image)
            plot_pixel(image)
            temp_ra.append(PSNR(image_matrix, image).execute())

            SCI = SimulationCompressingImage(image_matrix)
            image = SCI.oneside(raster, sampling_rate=sampling_list[i])
            image =  image
            image = rescale_2d_array(image)
            plot_pixel(image)
            temp_ras.append(PSNR(image_matrix, image).execute())
            print(temp_ha)

        hadamard_array = np.vstack((hadamard_array, np.array(temp_ha)))
        random_array = np.vstack((random_array, np.array(temp_ra)))
        raster_array = np.vstack((raster_array, np.array(temp_ras)))
    store_2d_array_to_csv(hadamard_array, "psnr_hadamard")
    store_2d_array_to_csv(random_array, "psnr_random")
    store_2d_array_to_csv(raster_array, "psnr_raster")
compare_psnr()
def store_2d_array_to_csv(array, file_name):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(array)


# Function to read data from a file
def read_data(file_path):
    return pd.read_csv(file_path, header=None)
def plot_PSNR():
    # File paths
    file_paths = {
        "hadamard": "psnr_hadamard",
        "random": "psnr_random",
        "raster": "psnr_raster"
    }

    # Reading the data and calculating mean and standard deviation for each column
    data_summary = {}
    for key, path in file_paths.items():
        data = read_data(path)
        mean_values = data.mean(axis=0)  # Calculating mean for each column
        std_values = data.std(axis=0)    # Calculating standard deviation for each column
        data_summary[key] = {
            "mean": mean_values,
            "std": std_values
        }

    # Sampling rates
    # Adjusting the x-axis to match the reversed order of sampling rates in the data

    # The sampling rates are reversed in the data files
    reversed_sampling_rates = [1, 0.5, 0.25, 0.1]

    # Plotting the data with the adjusted x-axis
    plt.figure(figsize=(10, 6))

    for key, summary in data_summary.items():
        # Filtering out NaN values in mean and standard deviation
        filtered_means = summary['mean'][~np.isnan(summary['mean']) & ~np.isnan(summary['std'])]
        filtered_stds = summary['std'][~np.isnan(summary['mean']) & ~np.isnan(summary['std'])]
        valid_sampling_rates = [reversed_sampling_rates[i] for i in filtered_means.index]

        plt.errorbar(valid_sampling_rates, filtered_means, yerr=filtered_stds, capsize=5, label=key.capitalize())

    plt.title('PSNR Values at Different Sampling Rates (Reversed Order)')
    plt.xlabel('Sampling Rate (Reversed)')
    plt.ylabel('PSNR Value')
    plt.legend()
    plt.grid(True)
    plt.gca().invert_xaxis()  # Inverting the x-axis to match the reversed order
    plt.show()

def test_fourier_plane():
    image_path = 'test_image.png'
    image = Image.open(image_path)
    image_resized = image.resize((128, 128))
    # Convert the image to grayscale
    image_gray = image_resized.convert('L')
    image_matrix = np.array(image_gray)
    fourier_transform = np.fft.fftshift(np.fft.fftn(image_matrix))
    plot_pixel(np.log(abs(fourier_transform)), "Image on Fourier Plane")

    call_pattern = generate_pattern.DmdPattern('hadamard', 128, 128, gray_scale=255)
    hadamard, negative_hadamard = call_pattern.execute(length=1)
    SCI = SimulationCompressingImage(image_matrix)
    image = SCI.fourier(hadamard, negative_hadamard, sampling_rate=1)
    image = image*-1
    image = np.log(abs(image))
    psn = PSNR(image_matrix, image).execute()
    plot_pixel(rescale_2d_array(image), f"PSNR={psn}")
def normalize_data(data):
    return   data/np.max(data)


def plot_metrics(number_of_measurements, psnr_data, mse_data, ssim_data):
    """
    Plot normalized PSNR, MSE, and SSIM over a range of measurements.

    :param number_of_measurements: List of measurement counts (x-axis).
    :param psnr_data: List of PSNR values.
    :param mse_data: List of MSE values.
    :param ssim_data: List of SSIM values.
    """
    # Normalize the data
    normalized_psnr = normalize_data(psnr_data)
    normalized_mse = normalize_data(mse_data)
    normalized_ssim = normalize_data(ssim_data)

    plt.figure(figsize=(10, 6))

    # Plotting the data
    plt.plot(number_of_measurements, normalized_psnr, label='Normalized PSNR', marker='o')
    plt.plot(number_of_measurements, normalized_mse, label='Normalized MSE', marker='s')
    plt.plot(number_of_measurements, normalized_ssim, label='Normalized SSIM', marker='^')

    plt.xlabel('Number of Measurements')
    plt.ylabel('Normalized Metric Value')
    plt.title('Comparison of Normalized PSNR, MSE, and SSIM')
    plt.legend()
    plt.grid(True)
    plt.show()
def image_with_different_sampling_rate():
    image_path = 'test_image.png'
    image = Image.open(image_path)
    image_resized = image.resize((128, 128))
    image_gray = image_resized.convert('L')
    image_matrix = np.array(image_gray)
    plot_pixel(image_matrix)
    call_pattern = generate_pattern.DmdPattern('hadamard', 128, 128, gray_scale=255)
    hadamard, negative_hadamard = call_pattern.execute(length=1)
    SCI = SimulationCompressingImage(image_matrix)
    image_list = SCI.image_with_different_sampling_rate(hadamard, negative_hadamard, num_image=31)
    plot_pixel(image_list[-1])
    SSIM_list = []
    MSE_list = []
    PSNR_list = []
    num_measurements = []
    for i in range(30):
        resized_image = rescale_2d_array(image_list[i])
        ini = PSNR(image_matrix, resized_image)
        SSIM_list.append(ini.SSIM())
        MSE_list.append(ini.mse())
        PSNR_list.append(ini.execute())
        num_measurements.append(int(128*128*(i+1)/30))
    plot_metrics(np.array(num_measurements), np.array(PSNR_list),np.array(MSE_list),np.array(SSIM_list))


def test_adaptive():
    image_path = 'ghost_image.png'
    image = Image.open(image_path)
    image_resized = image.resize((128, 128))
    # Convert the image to grayscale
    image_gray = image_resized.convert('L')
    image = np.array(image_gray)
    image = random_image_with_shapes(128,128, 5)
    ground_image = rescale_2d_array(image)
    plot_pixel(ground_image)
    a = SimulateAdaptiveCompressiveImage(2, 128, 1)
    dic = a.adaptive_pattern(ground_image)
    image = a.reconstruct_image_from_wavelet(dic)
    a, b, c = image
    plot_pixel(-1 * a)
    plot_pixel(b)
    plot_pixel(c)
    resized_a = cv2.resize(-1 * a, (128, 128), interpolation=cv2.INTER_LINEAR) + 250
    resized_b = cv2.resize(b, (128, 128), interpolation=cv2.INTER_LINEAR)
    final_image = rescale_2d_array( resized_b + c)
    plot_pixel(final_image)
    call_pattern = generate_pattern.DmdPattern('hadamard', 128, 128, gray_scale=255)
    hadamard, negative_hadamard = call_pattern.execute(length=1)
    SCI = SimulationCompressingImage(ground_image)
    image_list = SCI.image_with_different_sampling_rate(hadamard, negative_hadamard, num_image=50)
    SSIM_list = []
    PSNR_list = []
    num_measurements = []
    const_PSNR = []
    constants_SSIM = []
    groun = PSNR(ground_image, final_image).SSIM()
    groun_psnr = PSNR(ground_image, final_image).execute()
    for i in range(50):
        resized_image = rescale_2d_array(image_list[i])
        SSIM_list.append(PSNR(ground_image, resized_image).SSIM())
        PSNR_list.append(PSNR(ground_image,resized_image).execute())
        num_measurements.append(int(128*128*(i+1)/50))
        constants_SSIM.append(groun)
        const_PSNR.append(groun_psnr)

    print(PSNR_list)
    print(PSNR(ground_image, final_image).SSIM())
    plt.figure(figsize=(10, 6))

    # Plotting the data
    plt.plot(np.array(num_measurements), np.array(PSNR_list)/(np.max(np.array(PSNR_list))), label='CI PSNR', marker='o')
    plt.plot(np.array(num_measurements), np.array(SSIM_list), label='CI SSIM', marker='s')
    plt.plot(np.array(num_measurements), np.array(constants_SSIM), label='Adaptive Sampling SSIM', marker='s')

    plt.xlabel('Number of Measurements')
    plt.ylabel('Normalized Metric Value')
    plt.legend()
    plt.grid(True)
    plt.show()
def experiment_time_cost():
    data = {
        '16': [12.08017, 12.10335, 12.08097, 12.1005, 12.1067, 14.15529, 14.14455],
        '32': [14.22431, 14.32531, 14.22234, 14.25431, 14.63121, 14.21649, 14.24124],
        '64': [24.0842, 24.26395, 24.29851, 24.36901, 24.32973, 24.34446, 24.51627],
        '128': [120.8788, 127.9464, 125.3343, 127.8183, 129.7217, 127.2468, 130.6112]
    }

    # Calculate means and standard deviations
    means = np.array(list(map(np.mean, data.values())))
    stds = np.array(list(map(np.std, data.values())))
    resolutions = np.array(list(map(int, data.keys())))

    # Plotting
    fig, ax = plt.subplots()
    ax.errorbar(resolutions, means, yerr=stds, fmt='x', capsize=5, label='Mean ± STD', color="green")

    # Adding labels and title
    ax.set_xlabel('Resolution of Hadamard Mask')
    ax.set_ylabel('Time to Complete Task (s)')
    ax.set_title('Task Completion Time vs. Hadamard Mask Resolution')

    plt.show()
