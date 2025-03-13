import cv2
import numpy as np
import glob
from scipy.ndimage import median_filter, gaussian_filter, gaussian_filter1d
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import os

import matplotlib
matplotlib.use("TkAgg")

def covert2colorMap(img):

    normalized_imgg = img/np.mean(img)

    normalized_imgg[normalized_imgg >= 1.0] = 1.0

    gamma = 1
    normalized_imgg = np.power(normalized_imgg, gamma)

    normalized_imgg_uint8 = (normalized_imgg * 255).astype(np.uint8)

    colormap_imgg = cv2.applyColorMap(normalized_imgg_uint8, cv2.COLORMAP_JET)

    return colormap_imgg

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

def compute_envelope(image, image_90, FAST_MODE=True):


    amplitude_envelope = np.sqrt(image**2 + image_90**2)

    #no_signal_index = amplitude_envelope < 0.001

    #amplitude_envelope[no_signal_index] = 0

    

    return amplitude_envelope

def make_plot(value, image):

    import matplotlib
    matplotlib.use("TkAgg")

    plt.figure(figsize=(15, 7))
    plt.plot(np.arange(image.shape[0]), value, marker='o', linestyle='-', markersize=4)
    plt.title('Visualization of Image Column Values')
    plt.xlabel('Row Index')
    plt.ylabel('Pixel Value')
    plt.grid(True)
    

material = "gold_crown"
images_path = glob.glob(f"/media/piljae/X31/Dataset/Hubitz/depth_compute/original_images/gold_crown/4/divided_patterns/*.png")
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
divided_images = images[16:20]

sin_images = images[20:24]

L_d_list = []
L_g_list = []
max_list = []
min_list = []

rect_list = []
envelope_list = []


L_d_pattern_list = []
L_g_pattern_list = []

syn_patterns = []
original_patterns = []

for i in range(4):

    L_plus = np.maximum(divided_images[i], divided_images[(i+1)%4])
    L_minus = np.minimum.reduce([divided_images[i], divided_images[(i+1)%4]])

    cv2.imshow("L_plus", L_plus)
    cv2.imshow("L_minus", L_minus)
    cv2.waitKey(0)

    L_d = L_plus - L_minus
    L_g = 2 * L_minus

    L_d_pattern_list.append(L_d)
    L_g_pattern_list.append(L_g)

for i in range(4):

    image_ = L_d_pattern_list[i]
    image_inv_ = L_d_pattern_list[(i+2)%4]

    filtered_edge = filtering(image_ - image_inv_, parameter=1.0, type=1)

    alpha = 3
    syn_edge = sigmoid(alpha * filtered_edge)

    syn_patterns.append(syn_edge)

    cv2.imshow("syn_edge", syn_edge)
    cv2.waitKey(0)



    image_ = shift_0_images[i]
    image_inv_ = shift_0_images[(i+2)%4]

    filtered_edge = filtering(image_ - image_inv_, parameter=1.0, type=1)

    alpha = 3
    syn_edge = sigmoid(alpha * filtered_edge)

    original_patterns.append(syn_edge)


amplitude_maps = []
amplitude_maps_original = []

for i in range(4):

    ## syn original pattern amplitude map

    value = syn_patterns[i] - 0.5
    value2 = syn_patterns[(i + 1) % 4] - 0.5

    amplitude_envelope = np.sqrt(value**2 + value2**2)

    amplitude_maps.append(amplitude_envelope)


    v = original_patterns[i] - 0.5
    v2 = original_patterns[(i + 1) % 4] - 0.5

    amplitude_envelope_original = np.sqrt(v**2 + v2**2)

    amplitude_maps_original.append(amplitude_envelope_original)


    cv2.imshow("amplitude_envelope", amplitude_envelope)
    cv2.imshow("amplitude_envelope_color", covert2colorMap(amplitude_envelope))
    cv2.waitKey(0)


amplitude_sum_image = np.max(amplitude_maps, axis=0)

amplitude_sum_image_color = covert2colorMap(amplitude_sum_image)

cv2.imshow("amplitude_sum_image_color", amplitude_sum_image_color)

amplitude_maps_original

amplitude_maps_original_sum_image = np.max(amplitude_maps_original, axis=0)

amplitude_maps_original_sum_image_color = covert2colorMap(amplitude_maps_original_sum_image)

cv2.imshow("amplitude_maps_original_sum_image_color", amplitude_maps_original_sum_image_color)


gamma=0.5

image_corrected = np.power(amplitude_sum_image_color, gamma)  # 감마 적용

# 2. OpenCV는 0~255로 변환해야 제대로 저장됨
image_8bit = cv2.normalize(image_corrected, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

cv2.imwrite("/media/piljae/X31/experiment_result/original_hubitz_pattern/gold_crown/4/amplitude_map.png", image_8bit)
cv2.waitKey(0)
