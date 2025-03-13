import cv2
import numpy as np
import glob
from scipy.ndimage import median_filter, gaussian_filter, gaussian_filter1d
from scipy.signal import hilbert
import matplotlib.pyplot as plt

def my_hilbert(x):

    import matplotlib
    matplotlib.use("TkAgg")

    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    
    X = np.fft.fft(x, n=N)
    
    freq = np.fft.fftfreq(N, d=1.0)  # d=1.0은 샘플 간격 (예: 1초)

    magnitude = np.abs(X)

    index = magnitude < np.mean(X)

    X[index] = 0.0
    
    # plt.figure(figsize=(14, 6))
    # plt.stem(freq, np.abs(X), basefmt=" ")
    # plt.title('Original Signal FFT Spectrum (Positive Frequencies)')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Magnitude')
    # plt.grid(True)
    # plt.show()
    
    h = np.zeros(N)
    
    if N % 2 == 0:
        # 짝수 길이
        h[0] = 1          # DC 성분
        h[N//2] = 1       # Nyquist 주파수
        h[1:N//2] = 2     # 양의 주파수에 2배
    else:
        # 홀수 길이
        h[0] = 1          # DC 성분
        h[1:(N+1)//2] = 2 # 양의 주파수에 2배
    
    X_filtered = X * h
    
    # plt.figure(figsize=(14, 6))
    # plt.stem(freq[:N//2], np.abs(X_filtered)[:N//2], basefmt=" ", linefmt='r-', markerfmt='ro')
    # plt.title('FFT Spectrum after Applying Hilbert Mask (Positive Frequencies)')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Magnitude')
    # plt.grid(True)
    # plt.show()
    
    x_analytic = np.fft.ifft(X_filtered, n=N)
    
    # plt.figure(figsize=(14, 8))
    
    # plt.subplot(3, 1, 1)
    # plt.plot(x, label='Original Signal')
    # plt.title('Original Signal')
    # plt.xlabel('Sample')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.grid(True)
    
    # plt.subplot(3, 1, 2)
    # plt.plot(x_analytic.real, label='Real Part (Original Signal)', color='blue')
    # plt.plot(x_analytic.imag, label='Imaginary Part (Hilbert Transform)', color='orange')
    # plt.title('Analytic Signal Components')
    # plt.xlabel('Sample')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.grid(True)
    
    # plt.subplot(3, 1, 3)
    # amplitude_envelope = np.abs(x_analytic)
    # plt.plot(amplitude_envelope, label='Amplitude Envelope', color='green')
    # plt.plot(x, label='Original Signal', alpha=0.5)
    # plt.title('Amplitude Envelope')
    # plt.xlabel('Sample')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.grid(True)
    
    # plt.tight_layout()
    # plt.show()
    
    return x_analytic

def adaptive_alpha_scaling(image, image_90, FAST_MODE=True):

    if FAST_MODE:

        alpha_map = np.zeros_like(image)

        amplitude_envelope = np.sqrt(image**2 + image_90**2)

        alpha = 0.1/amplitude_envelope

        no_signal_index = amplitude_envelope < 0.00000001

        amplitude_envelope[no_signal_index] = 0
        alpha[no_signal_index] = 0

        alpha_map = alpha
        

        return alpha_map
    
    else:

        # import matplotlib
        # matplotlib.use("TkAgg")
        alpha_map = np.zeros_like(image)

        for i in range(image.shape[1]):

            value = image[:, i]

            analytic_signal = my_hilbert(value)
            
            #analytic_signal2 = my_hilbert(value)
            
            # real_part = np.real(analytic_signal)
            # imag_part = np.imag(analytic_signal)
            amplitude_envelope = np.abs(analytic_signal)

            alpha = 0.1/amplitude_envelope

            no_signal_index = amplitude_envelope < 0.00000001

            amplitude_envelope[no_signal_index] = 0
            alpha[no_signal_index] = 0

            alpha_map[:, i] = alpha
            

        return alpha_map

def adaptive_alpha_scaling_fast(image, image_90):

    alpha_map = np.zeros_like(image)

    amplitude_envelope = np.sqrt(image**2 + image_90**2)

    alpha = 0.1/amplitude_envelope

    no_signal_index = amplitude_envelope < 0.00000001

    amplitude_envelope[no_signal_index] = 0
    alpha[no_signal_index] = 0

    alpha_map = alpha
        

    return alpha_map

def filtering(image, parameter=3, type=0):
    
    if type == 0:
        return median_filter(image, size=parameter)
    
    if type == 1:
        return gaussian_filter(image, sigma=parameter)
    
    if type == 2:
        return gaussian_filter1d(image, sigma=parameter, axis=1)
    
    else:
        return image

# Sigmoid 함수
def sigmoid(x, alpha=1.0):
    return 1 / (1 + np.exp(-alpha * x))


def apply_gamma(image, max_value, gamma=1.0):

    corrected = np.power(image/max_value, gamma)
    return corrected

material = "gold_crown"
images_path = glob.glob(f"/media/piljae/X31/Dataset/Hubitz/depth_compute/original_images/{material}/1/*.png")
images_path.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

images = []
for image_path in images_path:
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    images.append(img)

images = np.array(images)

shift_0_images = images[:4]
low_0_images = images[4:8]
middle_0_images = images[8:12]
shift_45_images = images[12:16]
rgb_images = images[16:]

# L_plus = np.maximum.reduce([shift_0_images[0], shift_0_images[1], shift_0_images[2], shift_0_images[3]])
# L_minus = np.minimum.reduce([shift_0_images[0], shift_0_images[1], shift_0_images[2], shift_0_images[3]])
# L_d = L_plus - L_minus


# type = 3
# img_type = shift_45_images

# image = np.maximum.reduce([low_0_images[0], low_0_images[1], low_0_images[2], low_0_images[3]])

# for i in range(len(img_type)):
#     cv2.imshow("image_", img_type [i] )
#     cv2.waitKey(0)


# for i in range(len(img_type)):

import os

filtering_type = "none"

filterings = ["median", "gaussian", "none", "median_gaussian", "gaussian_gaussian"]

kernel_size = 3
gaussian_parameter = 1.0

FAST_MODE = False # fast alpha scaling

for i in range(5):

    if i == 4:
        
        type = i
        rgb_images = images[4 * i:4 * i + 5]

        cv2.imwrite(f"adaptive_alpha_direct_16_pattern/{4*type + 0}.png", (rgb_images[0] * 255).astype(np.uint8))
        cv2.imwrite(f"adaptive_alpha_direct_16_pattern/{4*type + 1}.png", (rgb_images[1] * 255).astype(np.uint8))
        cv2.imwrite(f"adaptive_alpha_direct_16_pattern/{4*type + 2}.png", (rgb_images[2] * 255).astype(np.uint8))
        cv2.imwrite(f"adaptive_alpha_direct_16_pattern/{4*type + 3}.png", (rgb_images[3] * 255).astype(np.uint8))
        cv2.imwrite(f"adaptive_alpha_direct_16_pattern/{4*type + 4}.png", (rgb_images[4] * 255).astype(np.uint8))

        for filtering_type in filterings:
            for alpha in alpha_list:
                for beta in beta_list:
                    
                    if filtering_type == "median":
                        folder_path = f"adaptive_alpha_direct_16_pattern/{filtering_type}_filtering_{kernel_size}x{kernel_size}/alpha_{alpha}_beta_{beta}"

                    else:
                        folder_path = f"adaptive_alpha_direct_16_pattern/{filtering_type}_filtering/alpha_{alpha}_beta_{beta}"
                    
                    os.makedirs(folder_path, exist_ok=True)

                    cv2.imwrite(f"{folder_path}/{4*type + 0}.png", (rgb_images[0] * 255).astype(np.uint8))
                    cv2.imwrite(f"{folder_path}/{4*type + 1}.png", (rgb_images[1] * 255).astype(np.uint8))
                    cv2.imwrite(f"{folder_path}/{4*type + 2}.png", (rgb_images[2] * 255).astype(np.uint8))
                    cv2.imwrite(f"{folder_path}/{4*type + 3}.png", (rgb_images[3] * 255).astype(np.uint8))
                    cv2.imwrite(f"{folder_path}/{4*type + 4}.png", (rgb_images[4] * 255).astype(np.uint8))


    else:
        type = i
        img_type = images[4*i : 4*i + 4]

        alpha_list = [i for i in range(0, 51, 5)]
        alpha_list.append(1)
        alpha_list.append(3)
        beta_list = [1]

        image1 = img_type[0]
        image2 = img_type[1]
        image3 = img_type[2]
        image4 = img_type[3]

        # cv2.imshow("image1:", image1)
        # cv2.imshow("image2:", image2)
        # cv2.imshow("image3:", image3)
        # cv2.imshow("image4:", image4)
        #cv2.waitKey(0)

        cv2.imwrite(f"adaptive_alpha_direct_16_pattern/{4*type + 0}.png", (image1 * 255).astype(np.uint8))
        cv2.imwrite(f"adaptive_alpha_direct_16_pattern/{4*type + 1}.png", (image2 * 255).astype(np.uint8))
        cv2.imwrite(f"adaptive_alpha_direct_16_pattern/{4*type + 2}.png", (image3 * 255).astype(np.uint8))
        cv2.imwrite(f"adaptive_alpha_direct_16_pattern/{4*type + 3}.png", (image4 * 255).astype(np.uint8))

        
        max_image = np.maximum.reduce([image1, image2, image3, image4])
        min_image = np.minimum.reduce([image1, image2, image3, image4])

        L_d = max_image - min_image

        L_g = 0.5 * min_image

        # cv2.imshow("L_d:", L_d)
        # cv2.imshow("max_image:", max_image)
        # cv2.imshow("min_image:", min_image * 0.5)
        # cv2.imshow("image", L_d + L_g)
        # cv2.waitKey(0)

        # cv2.imwrite("direct_pattern/max_image.png", (max_image * 255).astype(np.uint8))
        # cv2.imwrite("direct_pattern/min_image.png", (min_image * 255).astype(np.uint8))
        # cv2.imwrite("direct_pattern/L_d.png", (L_d * 255).astype(np.uint8))
        # cv2.imwrite("direct_pattern/L_g.png", (L_g * 255).astype(np.uint8))
        # cv2.waitKey(0)


        for filtering_type in filterings:
            for alpha in alpha_list:
                for beta in beta_list:


                    if filtering_type == "median":
                        folder_path = f"adaptive_alpha_direct_16_pattern/{filtering_type}_filtering_{kernel_size}x{kernel_size}/alpha_{alpha}_beta_{beta}"

                    else:
                        folder_path = f"adaptive_alpha_direct_16_pattern/{filtering_type}_filtering/alpha_{alpha}_beta_{beta}"
                    
                    os.makedirs(folder_path, exist_ok=True)
                    
                    w_med = sigmoid(alpha * filtering(image1 - image3, parameter=kernel_size), alpha=beta)
                    w_gaus = sigmoid(alpha * filtering(image1 - image3, parameter=gaussian_parameter, type=1), alpha=beta)
                    w_none = sigmoid(alpha * filtering(image1 - image3, parameter=gaussian_parameter, type=2), alpha=beta)
                    
                    # if alpha == 1 or alpha == 5 or alpha == 10 or alpha == 20 or alpha == 29:
                    #     # cv2.imwrite(f"median_{alpha}.png", w_med*255)
                    #     # cv2.imwrite(f"gauss_{alpha}.png", w_gaus*255)
                    #     # cv2.imwrite(f"none_{alpha}.png", w_none*255)


                    if filtering_type == "median":
                        #median filtering

                        filtering_1 = filtering(image1 - image3, parameter=kernel_size)
                        filtering_2 = filtering(image2 - image4, parameter=kernel_size)
                        filtering_3 = filtering(image3 - image1, parameter=kernel_size)
                        filtering_4 = filtering(image4 - image2, parameter=kernel_size)

                        w_1 = sigmoid(alpha * adaptive_alpha_scaling(filtering_1, filtering_2, FAST_MODE) * filtering_1, alpha=beta)
                        w_2 = sigmoid(alpha * adaptive_alpha_scaling(filtering_2, filtering_3, FAST_MODE) * filtering_2, alpha=beta)
                        w_3 = sigmoid(alpha * adaptive_alpha_scaling(filtering_3, filtering_4, FAST_MODE) * filtering_3, alpha=beta)
                        w_4 = sigmoid(alpha * adaptive_alpha_scaling(filtering_4, filtering_1, FAST_MODE) * filtering_4, alpha=beta)

                    elif filtering_type == "gaussian":
                        # gaussian filtering

                        filtering_1 = filtering(image1 - image3, parameter=gaussian_parameter, type=1)
                        filtering_2 = filtering(image2 - image4, parameter=gaussian_parameter, type=1)
                        filtering_3 = filtering(image3 - image1, parameter=gaussian_parameter, type=1)
                        filtering_4 = filtering(image4 - image2, parameter=gaussian_parameter, type=1)


                        w_1 = sigmoid(alpha * adaptive_alpha_scaling(filtering_1, filtering_2, FAST_MODE) * filtering_1, alpha=beta)
                        w_2 = sigmoid(alpha * adaptive_alpha_scaling(filtering_2, filtering_3, FAST_MODE) * filtering_2, alpha=beta)
                        w_3 = sigmoid(alpha * adaptive_alpha_scaling(filtering_3, filtering_4, FAST_MODE) * filtering_3, alpha=beta)
                        w_4 = sigmoid(alpha * adaptive_alpha_scaling(filtering_4, filtering_1, FAST_MODE) * filtering_4, alpha=beta)
                        
                    elif filtering_type == "gaussian_gaussian":
                        # gaussian filtering

                        filtering_1 = filtering(image1 - image3, parameter=gaussian_parameter, type=1)
                        filtering_2 = filtering(image2 - image4, parameter=gaussian_parameter, type=1)
                        filtering_3 = filtering(image3 - image1, parameter=gaussian_parameter, type=1)
                        filtering_4 = filtering(image4 - image2, parameter=gaussian_parameter, type=1)
                        
                        alpha_map_filtering_1 = filtering(adaptive_alpha_scaling(filtering_1, filtering_2, FAST_MODE), parameter=gaussian_parameter, type=1)
                        alpha_map_filtering_2 = filtering(adaptive_alpha_scaling(filtering_2, filtering_3, FAST_MODE), parameter=gaussian_parameter, type=1)
                        alpha_map_filtering_3 = filtering(adaptive_alpha_scaling(filtering_3, filtering_4, FAST_MODE), parameter=gaussian_parameter, type=1)
                        alpha_map_filtering_4 = filtering(adaptive_alpha_scaling(filtering_4, filtering_1, FAST_MODE), parameter=gaussian_parameter, type=1)
                    
                        w_1 = sigmoid(alpha * alpha_map_filtering_1 * filtering_1, alpha=beta)
                        w_2 = sigmoid(alpha * alpha_map_filtering_2 * filtering_2, alpha=beta)
                        w_3 = sigmoid(alpha * alpha_map_filtering_3 * filtering_3, alpha=beta)
                        w_4 = sigmoid(alpha * alpha_map_filtering_4 * filtering_4, alpha=beta)               
                    
                    elif filtering_type == "median_gaussian":
                        # gaussian filtering

                        filtering_1 = filtering(image1 - image3, parameter=kernel_size, type=0)
                        filtering_2 = filtering(image2 - image4, parameter=kernel_size, type=0)
                        filtering_3 = filtering(image3 - image1, parameter=kernel_size, type=0)
                        filtering_4 = filtering(image4 - image2, parameter=kernel_size, type=0)

                        filtering_1 = filtering(filtering_1, parameter=gaussian_parameter, type=1)
                        filtering_2 = filtering(filtering_2, parameter=gaussian_parameter, type=1)
                        filtering_3 = filtering(filtering_3, parameter=gaussian_parameter, type=1)
                        filtering_4 = filtering(filtering_4, parameter=gaussian_parameter, type=1)
                    
                        w_1 = sigmoid(alpha * adaptive_alpha_scaling(filtering_1, filtering_2, FAST_MODE) * filtering_1, alpha=beta)
                        w_2 = sigmoid(alpha * adaptive_alpha_scaling(filtering_2, filtering_3, FAST_MODE) * filtering_2, alpha=beta)
                        w_3 = sigmoid(alpha * adaptive_alpha_scaling(filtering_3, filtering_4, FAST_MODE) * filtering_3, alpha=beta)
                        w_4 = sigmoid(alpha * adaptive_alpha_scaling(filtering_4, filtering_1, FAST_MODE) * filtering_4, alpha=beta)   
                    
                    else:
                        # none filtering
                        w_1 = sigmoid(alpha * adaptive_alpha_scaling(image1 - image3, image2 - image4, FAST_MODE) * filtering(image1 - image3, parameter=1.0, type=3), alpha=beta)
                        w_2 = sigmoid(alpha * adaptive_alpha_scaling(image2 - image4, image3 - image1, FAST_MODE) * filtering(image2 - image4, parameter=1.0, type=3), alpha=beta)
                        w_3 = sigmoid(alpha * adaptive_alpha_scaling(image3 - image1, image4 - image2, FAST_MODE) * filtering(image3 - image1, parameter=1.0, type=3), alpha=beta)
                        w_4 = sigmoid(alpha * adaptive_alpha_scaling(image4 - image2, image1 - image3, FAST_MODE) * filtering(image4 - image2, parameter=1.0, type=3), alpha=beta)

                    # cv2.imshow("w_1:", w_1)
                    # cv2.imshow("w_2:", w_2)
                    # cv2.imshow("w_3:", w_3)
                    # cv2.imshow("w_4:", w_4)
                    # cv2.waitKey(0)
                    
                    #index = max_image < 0.03
                    L_d = np.ones_like(L_d)


                    #L_d[index] = 0

                    I1_ours = w_1 * L_d
                    I2_ours = w_2 * L_d
                    I3_ours = w_3 * L_d
                    I4_ours = w_4 * L_d

                    max_value1 = np.max(I1_ours)
                    max_value2 = np.max(I2_ours)
                    max_value3 = np.max(I3_ours)
                    max_value4 = np.max(I4_ours)

                    I1_ours = I1_ours/max_value1
                    I2_ours = I2_ours/max_value2
                    I3_ours = I3_ours/max_value3
                    I4_ours = I4_ours/max_value4

                    # I1_ours = np.clip(I1_ours * 2 , 0, 1)
                    # I2_ours = np.clip(I2_ours * 2 , 0, 1)
                    # I3_ours = np.clip(I3_ours * 2 , 0, 1)
                    # I4_ours = np.clip(I4_ours * 2 , 0, 1)
                    
                    cv2.imwrite(f"{folder_path}/{4*type + 0}.png", (I1_ours * 255).astype(np.uint8))
                    cv2.imwrite(f"{folder_path}/{4*type + 1}.png", (I2_ours * 255).astype(np.uint8))
                    cv2.imwrite(f"{folder_path}/{4*type + 2}.png", (I3_ours * 255).astype(np.uint8))
                    cv2.imwrite(f"{folder_path}/{4*type + 3}.png", (I4_ours * 255).astype(np.uint8))
                    
                    print(folder_path)

                    # if filtering_type == "gaussian" and alpha == 50:
                    #     cv2.imshow("verify:", w_1)
                    #     cv2.imshow("I1_ours:", I1_ours)
                    #     cv2.imshow("I2_ours:", I2_ours)
                    #     cv2.imshow("I3_ours:", I3_ours)
                    #     cv2.imshow("I4_ours:", I4_ours)
                    #     cv2.waitKey(0)