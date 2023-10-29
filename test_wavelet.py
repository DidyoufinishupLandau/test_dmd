import pywt
def wavelet_transform_level_one(matrix):
    # Apply single level Discrete Wavelet Transform
    coeffs = pywt.dwt2(matrix, 'db1')  # 'db1' refers to the Daubechies wavelet with one vanishing moment.
    cA, (cH, cV, cD) = coeffs  # cA: approximation, cH: horizontal details, cV: vertical details, cD: diagonal details

    return cA, cH, cV, cD
def inverse_wavelet_trasform(cA, cH, cV, cD):
    reconstructed_data_2D = pywt.idwt2((cA, (cH, cV, cD)), 'db1')
    return reconstructed_data_2D
