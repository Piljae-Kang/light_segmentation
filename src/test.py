import cv2
import numpy as np
import glob
from scipy.ndimage import median_filter, gaussian_filter, gaussian_filter1d
import matplotlib.pyplot as plt
from scipy.signal import hilbert

def my_hilbert2(x):

    import matplotlib
    matplotlib.use("TkAgg")

    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    
    X = np.fft.fft(x, n=N)
    
    freq = np.fft.fftfreq(N, d=1.0)  # d=1.0은 샘플 간격 (예: 1초)

    magnitude = np.abs(X)
    
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
    
    
    x_analytic = np.fft.ifft(X_filtered, n=N)
    
    
    return x_analytic

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

def show_pixel_value(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  # 마우스가 움직일 때
        pixel_value = img[y, x]  # (y, x) 위치의 픽셀 값
        print(f"Position: ({y}, {x}), Pixel Value: {pixel_value}")

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


def adaptive_alpha_scaling_(image, image_inv):

    import matplotlib
    matplotlib.use("TkAgg")

    alpha_map = np.zeros_like(image)
    new_image = np.zeros_like(image)
    confidence_map = np.zeros_like(image)

    for i in range(image.shape[1]):

        value = image[:, i]

        analytic_signal = hilbert(value)
        # real_part = np.real(analytic_signal)
        # imag_part = np.imag(analytic_signal)

        value2 = image_inv[:, i]
        total_value = np.sqrt(value**2 + value2**2)
        
        amplitude_envelope_origin = np.abs(analytic_signal)
        amplitude_envelope = total_value
        #amplitude_envelope = amplitude_envelope_origin

        alpha = 0.1/amplitude_envelope

        no_signal_index = amplitude_envelope < 0.00000001

        amplitude_envelope[no_signal_index] = 0
        alpha[no_signal_index] = 0
        #alpha_map[:, i] = alpha
        #new_image[:, i] = analytic_signal.real

        confidence_map[:, i] = amplitude_envelope

        masked_signal = amplitude_envelope.copy()
        masked_signal[amplitude_envelope < 0.03] = 0

        phase = np.unwrap(np.angle(analytic_signal))
        phase2 = np.unwrap(np.arctan2(value, value2))
        grad_phase = np.gradient(phase)

        grad_index = grad_phase < 1

        grad_phase[grad_index] = 0

        masked_signal[grad_index] = 0.1
        confidence_map[:, i] = masked_signal

        alpha[grad_index] = 1

        alpha_map[:, i] = alpha

        amplitude_envelope = total_value
        amplitude_envelope2 = amplitude_envelope_origin

        alpha = 0.1/amplitude_envelope
        alpha2 = 0.1/amplitude_envelope2

        no_signal_index = amplitude_envelope < 0.000001

        amplitude_envelope[no_signal_index] = 0
        alpha[no_signal_index] = 0
        alpha2[no_signal_index] = 0

        alpha_map[:, i] = alpha


        if i == 200:

            plt.figure(figsize=(16, 12))
            plt.subplot(2, 1, 1)
            #plt.plot(np.arange(range_1), value0, marker='o', linestyle='-', markersize=4)
            plt.plot(np.arange(range_1), value, marker='o', linestyle='-', markersize=4)
            # plt.plot(np.arange(range_1), value2, marker='o', linestyle='-', markersize=4)
            plt.plot(np.arange(range_1), amplitude_envelope, label='Amplitude Envelope', linestyle='--')
            plt.plot(np.arange(range_1), amplitude_envelope2, label='Amplitude Envelope', linestyle='--')
            #plt.plot(np.arange(range_1), value2, label='Amplitude total_value', linestyle='--')
            #plt.plot(np.arange(range_1), total_value, label='Amplitude total_value', linestyle='--')
            # plt.plot(np.arange(range_1), alpha, label='alpha', linestyle='--')
            # plt.plot(np.arange(range_1), alpha2, label='alpha', linestyle='--')
            # plt.plot(t, real_part, label='Real Part (Original)', alpha=0.5)
            # plt.plot(t, imag_part, label='Imaginary Part (Hilbert Transform)', alpha=0.7)
            plt.title('Visualization of Image Column Values')
            plt.xlabel('Row Index')
            plt.ylabel('Pixel Value')
            
            plt.subplot(2, 1, 2)
            #plt.plot(np.arange(image.shape[0]), alpha, label='Amplitude Envelope', linestyle='--')
            plt.plot(np.arange(range_1), alpha, label='alpha', linestyle='--')
            #plt.plot(np.arange(range_1), alpha2, label='alpha', linestyle='--')
            #plt.plot(np.arange(range_1), phase, label='alpha', linestyle='--')
            #plt.plot(np.arange(range_1), phase, label='alpha', linestyle='--')
            #plt.plot(np.arange(image.shape[0]), grad_phase, label='grad_phase', linestyle='--')
            #plt.plot(np.arange(image.shape[0]), masked_signal, label='masked_signal', linestyle='--')
            plt.title('Visualization of Image Column Phase')
            plt.xlabel('Row Index')
            plt.ylabel('Phase')

            plt.show()

    #cv2.imshow("new_image", new_image)
    #cv2.imshow("image", image)
    cv2.imshow("confidence_map", confidence_map/np.max(confidence_map))
    cv2.imshow("confidence_maps", confidence_map)
    cv2.imshow("alpha_map", alpha_map)
    cv2.waitKey(0)

    return alpha_map, new_image

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
    
def make_plot(value, image):

    import matplotlib
    matplotlib.use("TkAgg")

    plt.figure(figsize=(5, 5))
    plt.plot(np.arange(image.shape[0]), value, marker='o', linestyle='-', markersize=4)
    plt.title('Visualization of Image Column Values')
    plt.xlabel('Row Index')
    plt.ylabel('Pixel Value')
    plt.grid(True)
    plt.show()

def analyze_patch(patch_center, shift_0_images, low_0_images):

    import matplotlib
    matplotlib.use("TkAgg")
    plt.figure(figsize=(5, 5))

    for h_image in shift_0_images:

        patch = h_image[:, patch_center[1]]
        plt.plot(np.arange(len(patch)), patch, marker='o', linestyle='-', markersize=4)

    plt.show()

    for l_image in low_0_images:

        patch = l_image[:, patch_center[1]]
        plt.plot(np.arange(len(patch)), patch, marker='o', linestyle='-', markersize=4)

    plt.show()

    


material = "gold_crown"
images_path = glob.glob(f"/media/piljae/X31/Dataset/Hubitz/depth_compute/original_images/{material}/1/*.png")
images_path.sort()

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


image = low_0_images[1]
image_inv = low_0_images[3]

max = np.maximum(image, image_inv)
min = np.minimum(image, image_inv)

# image_repeart = np.repeat(image.reshape(image.shape[0], image.shape[1], 1), 3, axis=-1)

# # 마우스 이벤트 콜백 함수
# def mouse_callback(event, x, y, flags, param):
#     if event == cv2.EVENT_MOUSEMOVE:  # 마우스 움직일 때 좌표 출력
#         print(f"Cursor Position: ({x}, {y})")

# # OpenCV 윈도우 생성 및 마우스 콜백 등록
# cv2.namedWindow("Image")
# cv2.setMouseCallback("Image", mouse_callback)

# while True:
#     cv2.imshow("Image", image_repeart)
#     if cv2.waitKey(1) & 0xFF == 27:  # ESC 키를 누르면 종료
#         break


# metal_crd = (290, 200)
# teeth_crd = (290, 400)


# for image in shift_0_images:
#     print(image[metal_crd])
# print("-------------------")
# for image in low_0_images:
#     print(image[metal_crd])

# print("#####################")
# for image in shift_0_images:
#     print(image[teeth_crd])
# print("-------------------")
# for image in low_0_images:
#     print(image[teeth_crd])

# breakpoint()

# analyze_patch(metal_crd, shift_0_images, low_0_images)
# analyze_patch(teeth_crd, shift_0_images, low_0_images)

############################# direct / indirect image ######################################


L_d_list = []
L_g_list = []
max_list = []
min_list = []

for i in range(4):

    image1 = images[4*i]
    image2 = images[4*i+1]
    image3 = images[4*i+2]
    image4 = images[4*i+3]

    min = np.minimum.reduce([image1, image2, image3, image4])
    max = np.maximum.reduce([image1, image2, image3, image4])
    
    # min = np.minimum.reduce([image1, image3])
    # max = np.maximum.reduce([image1, image3])

    L_d = max - min
    L_g = 0.5*min

    # Intesity-based reflection detection
    mask1 = L_g / L_d  > 2.0    # if the global illumination is two-times stronger than the direct illumination, true
    mask2 = ( L_d - L_g ) < -0.03 # if the global illumination is slightly stronger than the direction illumination, true
    mask_img = np.zeros_like(images[0])
    mask = mask1 & mask2
    mask_img[mask] = 1
    cv2.imshow(f"mask_{i}", mask_img)

    mask_img = np.zeros_like(images[0])
    mask = mask1 
    mask_img[mask] = 1
    cv2.imshow(f"mask_{i}1", mask_img)

    mask_img = np.zeros_like(images[0])
    mask =  mask2
    mask_img[mask] = 1
    cv2.imshow(f"mask_{i}2", mask_img)


    cv2.imshow(f"L_d_{i}", L_d)
    cv2.imshow(f"L_g_{i}", L_g)
    cv2.waitKey(0)






    L_d_list.append(L_d)
    L_g_list.append(L_g)
    max_list.append(max)
    min_list.append(min)

cv2.imshow("high", max_list[0])
cv2.imshow("low", max_list[1])
cv2.waitKey(0)


mask = (max_list[0] - max_list[1]) > -0.0

#imgg = L_d_list[1] + 2 * L_g_list[1] - L_d_list[0] - 2 * L_g_list[0]

imgg = min_list[1] - min_list[0]
min_imgg = np.min(imgg)

if min_imgg < 0:
    imgg -= min_imgg

normalized_imgg = imgg/np.max(imgg)
normalized_imgg_uint8 = (normalized_imgg * 255).astype(np.uint8)

colormap_imgg = cv2.applyColorMap(normalized_imgg_uint8, cv2.COLORMAP_JET)

cv2.imshow("colormap", colormap_imgg)


mask_img = np.zeros_like(images[0])
mask_img[mask] = 1
    
cv2.imshow("diff", (L_d_list[0] + 2 * L_g_list[0] - L_d_list[1] - 2 * L_g_list[1]) )
cv2.imshow("mask_img", mask_img)
cv2.waitKey(0)

for i in range(4):

    cv2.imshow("ld-h_g", images[4 + i] - L_g_list[0] - L_g_list[0])
    images[4 + i] = images[4 + i] - 5 * L_g_list[0]
    cv2.waitKey(0)

gaussian_parameter = 1.0
FAST_MODE = True
beta = 1.0
alpha = 20.0

for i in range(4):

    image1 = images[4*i]
    image2 = images[4*i+1]
    image3 = images[4*i+2]
    image4 = images[4*i+3]

    filtering_1 = filtering(image1 - image3, parameter=gaussian_parameter, type=1)
    filtering_2 = filtering(image2 - image4, parameter=gaussian_parameter, type=1)
    filtering_3 = filtering(image3 - image1, parameter=gaussian_parameter, type=1)
    filtering_4 = filtering(image4 - image2, parameter=gaussian_parameter, type=1)


    w_1 = sigmoid(alpha * adaptive_alpha_scaling(filtering_1, filtering_2, FAST_MODE) * filtering_1, alpha=beta)
    w_2 = sigmoid(alpha * adaptive_alpha_scaling(filtering_2, filtering_3, FAST_MODE) * filtering_2, alpha=beta)
    w_3 = sigmoid(alpha * adaptive_alpha_scaling(filtering_3, filtering_4, FAST_MODE) * filtering_3, alpha=beta)
    w_4 = sigmoid(alpha * adaptive_alpha_scaling(filtering_4, filtering_1, FAST_MODE) * filtering_4, alpha=beta)

    w_1 = w_1/np.max(w_1)
    w_2 = w_2/np.max(w_2)
    w_3 = w_3/np.max(w_3)
    w_4 = w_4/np.max(w_4)


    cv2.imshow("w_1", w_1)
    cv2.imshow("w_2", w_2)
    cv2.imshow("w_3", w_3)
    cv2.imshow("w_4", w_4)
    cv2.waitKey(0)

################################################################################


# alpha_map_gaussian, new_gaussian_image = adaptive_alpha_scaling(gaussian_image1, gaussian_image2)
# cv2.imshow("alpha_map_gaussian", alpha_map_gaussian)
# cv2.waitKey(0)

# adaptive_alpha_scaling_w_1 = sigmoid(20 * alpha_map_gaussian * gaussian_image3)
# adaptive_alpha_scaling_ = 20 * alpha_map_gaussian * gaussian_image1
# max_alpha = np.max(adaptive_alpha_scaling_w_1)

# cv2.imshow("adaptive_alpha_scaling_w_1", adaptive_alpha_scaling_w_1)
# #cv2.imshow("alpha_map_gaussian", alpha_map_gaussian/max_alpha)
# cv2.waitKey(0)

# make_plot(gaussian_image1[:, 150], gaussian_image1)
# make_plot(adaptive_alpha_scaling_[:, 150], adaptive_alpha_scaling_)
# make_plot(adaptive_alpha_scaling_w_1[:, 150], adaptive_alpha_scaling_w_1)


# max_alpha = np.max(alpha_map_gaussian)
# normalized_image = (alpha_map_gaussian/max_alpha * 255).astype(np.uint8)

# # 2. 컬러맵 적용 (COLORMAP_JET 사용)
# colormap = cv2.applyColorMap(normalized_image, cv2.COLORMAP_JET)

# cv2.imshow("colormap", colormap)

# adaptive_alpha_scaling_w_1_filtering = filtering(adaptive_alpha_scaling_w_1, parameter=3, type=0)
# cv2.imshow("adaptive_alpha_scaling_w_1_filtering", adaptive_alpha_scaling_w_1_filtering)
# cv2.waitKey(0)