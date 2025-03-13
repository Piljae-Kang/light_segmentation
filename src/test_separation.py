import cv2
import numpy as np
import glob
from scipy.ndimage import median_filter, gaussian_filter, gaussian_filter1d
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import os

import matplotlib
matplotlib.use("TkAgg")

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

def make_plot(value, image):

    import matplotlib
    matplotlib.use("TkAgg")

    analytic_signal = hilbert(value)
    amplitude_envelope = np.abs(analytic_signal)

    plt.figure(figsize=(15, 5))
    #plt.plot(np.arange(image.shape[0]), value, marker='o', linestyle='-', markersize=4)
    plt.plot(np.arange(image.shape[0]), amplitude_envelope, marker='o', linestyle='-', markersize=4)
    plt.title('Visualization of Image Column Values')
    plt.xlabel('Row Index')
    plt.ylabel('Pixel Value')
    plt.grid(True)
    # plt.show()


def covert2colorMap(img):

    normalized_imgg = img/np.mean(img) * 1.2

    normalized_imgg[normalized_imgg >= 1.0] = 1.0

    gamma = 1
    normalized_imgg = np.power(normalized_imgg, gamma)

    normalized_imgg_uint8 = (normalized_imgg * 255).astype(np.uint8)

    colormap_imgg = cv2.applyColorMap(normalized_imgg_uint8, cv2.COLORMAP_JET)

    return colormap_imgg


def save_img(path, img, gap, idx):

    gamma=0.5

    image_corrected = np.power(img, gamma)  # 감마 적용

    # 2. OpenCV는 0~255로 변환해야 제대로 저장됨
    image_8bit = cv2.normalize(image_corrected, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    if gap == 0:
        gap_num = 8
    elif gap == 1:
        gap_num = 12
    elif gap == 2:
        gap_num = 20
    else:
        gap_num = -1

    folder_path = f"{path}/gap_{gap_num}"

    os.makedirs(folder_path, exist_ok=True)
    cv2.imwrite(f"{folder_path}/{idx}.png", image_8bit)

# def mouse_callback(event, x, y, flags, param):
#     # 마우스가 움직일 때마다(MOUSEMOVE) 픽셀 값을 확인
#     if event == cv2.EVENT_MOUSEMOVE:
#         # B, G, R 순서로 픽셀이 저장되어 있음
#         b = img[y, x, 0]
#         print(f"X:{x}, Y:{y}, B:{b}")

material = "gold_crown"
case_num = 1
# images_path = glob.glob(f"/media/piljae/X31/Dataset/Hubitz/devided_pattern/{material}/frame/*.png")
images_path = glob.glob(f"/media/piljae/X31/Dataset/Hubitz/depth_compute/original_images/{material}/8pattern2/{case_num}/*.png")
images_path.sort()

date = "25_02_27"
base_path = f"/media/piljae/X31/experiment_result/ghost_pattern_detection/{material}/{date}/case_num_{case_num}"

images = []
for image_path in images_path:
    
    img = cv2.imread(image_path)
    img = img.astype(np.float32) / 255.0
    images.append(img)


images = np.array(images)

gap8_images = images[:12]
gap12_images = images[12:24]
gap20_images = images[24:36]

syn_pattern_list = []
syn_original_pattern_list = []
filtered_original_diff_patterns_list = []
for gap_num in range(3):

    imgs = images[12*gap_num : 12*(gap_num+1)]

    gap8_images = gap12_images

    syn_patterns = []

    syn_original_patterns = []

    filtered_original_diff_patterns = []

    for i in range(8):

        ascending_max = np.maximum.reduce([imgs[(0 + i) % 8], imgs[(1 + i) % 8], imgs[(2 + i) % 8], imgs[(3 + i) % 8]])
        ascending_min = np.minimum.reduce([imgs[(0 + i) % 8], imgs[(1 + i) % 8], imgs[(2 + i) % 8], imgs[(3 + i) % 8]])
        descending_max = np.maximum.reduce([imgs[(4 + i) % 8], imgs[(5 + i) % 8], imgs[(6 + i) % 8], imgs[(7 + i) % 8]])
        descending_min = np.minimum.reduce([imgs[(4 + i) % 8], imgs[(5 + i) % 8], imgs[(6 + i) % 8], imgs[(7 + i) % 8]])

        # cv2.imshow("gap8_pattern1_ascending_min",gap8_pattern1_ascending_min)
        # cv2.imshow("gap8_pattern1_descending_min", gap8_pattern1_descending_min)
        # cv2.waitKey(0)

        ascending_L_d = ascending_max - ascending_min
        ascending_L_g = 4 * descending_min

        #gap8_ascending_L_direct = gap8_images[8] - gap8_ascending_L_g

        descending_L_d = descending_max - descending_min
        descending_L_g = 4 * descending_min

        #gap8_descending_L_direct = gap8_images[9] - gap8_descending_L_g



        ascending = imgs[0] + imgs[1] + imgs[2] + imgs[3]
        descending = imgs[4] + gap8_images[5] + imgs[6] +  imgs[7]

        edge = ascending - descending

        edge = ( edge - np.min(edge) ) / np.max( edge - np.min(edge) )

        edge1 = ascending_L_d - descending_L_d

        edge1 = ( edge1 - np.min(edge1) ) / np.max( edge1 - np.min(edge1) )

        # cv2.imshow("edge", edge)
        # cv2.imshow("gap8_pattern1_ascending_max", gap8_pattern1_ascending_max)
        # cv2.imshow("gap8_ascending_L_d", 5 * gap8_ascending_L_d)
        # cv2.imshow("gap8_descending_L_d", 5 * gap8_descending_L_d)
        # cv2.imshow("gap8_ascending_L_direct", 5 * gap8_ascending_L_direct)
        # cv2.imshow("gap8_descending_L_direct", 5 * gap8_descending_L_direct)
        # cv2.imshow("edge1", edge1)
        # cv2.waitKey(0)

        if gap_num == 1:
            cv2.imwrite(f"direct_{i}.png", (ascending_L_d * 255).astype(np.uint8))
        
        cv2.imshow("ascending_L_d", ascending_L_d)
        cv2.waitKey(0)

        #filtered_edge = filtering(gap8_pattern1_ascending - gap8_pattern1_descending, parameter=1.0, type=1)
        filtered_edge = filtering(ascending_L_d - descending_L_d, parameter=1.0, type=1)

        alpha = 5
        syn_edge = sigmoid(alpha * filtered_edge)

        # cv2.imshow("syn_edge", syn_edge)
        # cv2.waitKey(0)

        # cv2.imshow("syn_edge_original", syn_edge_original)
        # cv2.waitKey(0)

        syn_patterns.append(syn_edge)


    syn_pattern_list.append(syn_patterns)

    for j in range(4):

        filtered_pattern_original = filtering(imgs[8 + j] - imgs[8 + ((j+2)%4)], parameter=1.0, type=1)
        syn_pattern_original = sigmoid(alpha * filtered_pattern_original)


        syn_original_patterns.append(syn_pattern_original)
        filtered_original_diff_patterns.append(filtering(imgs[8 + j], parameter=1.0, type=1))

    syn_original_pattern_list.append(syn_original_patterns)
    filtered_original_diff_patterns_list.append(filtered_original_diff_patterns)


####### image show pattern list ####################################

for gap, syn_patterns in enumerate(syn_pattern_list):

    for idx, syn_pattern in enumerate(syn_patterns):

        cv2.imshow("syn_pattern", syn_pattern)
        cv2.waitKey(0)

        save_img(f"{base_path}/syn_direct_pattern/", syn_pattern, gap, idx)

for gap, syn_original_patterns in enumerate(syn_original_pattern_list):

    for idx, syn_original_patterns in enumerate(syn_original_patterns):

        cv2.imshow("syn_original_patterns", syn_original_patterns)
        cv2.waitKey(0)

        save_img(f"{base_path}/syn_original_patterns/", syn_original_patterns, gap, idx)

#####################################################################


######################### compute amplitude map #####################

amplitude_maps_list = []
amplitude_original_maps_list = []
amplitude_colormaps_list = []
amplitude_original_colormaps_list = []

amplitude_natural_maps_list = []
amplitude_natural_colormaps_list = []

for gap_num in range(3):

    amplitude_maps = []
    amplitude_original_maps = []
    amplitude_colormaps = []
    amplitude_original_colormaps = []
    amplitude_natural_maps = []
    amplitude_natural_colormaps = []

    for i in range(8):

        value = syn_pattern_list[gap_num][i] - 0.5
        value2 = syn_pattern_list[gap_num][(i + 2) % 8] - 0.5

        amplitude_envelope = np.sqrt(value**2 + value2**2)

        #amplitude_envelope [amplitude_envelope < 0.01] = 0

        amplitude_maps.append(amplitude_envelope)
        amplitude_colormaps.append(covert2colorMap(amplitude_envelope))


    for j in range(4):

        ## syn original pattern amplitude map

        value = syn_original_pattern_list[gap_num][j] - 0.5
        value2 = syn_original_pattern_list[gap_num][(j + 1) % 4] - 0.5

        amplitude_envelope = np.sqrt(value**2 + value2**2)

        amplitude_original_maps.append(amplitude_envelope)
        amplitude_original_colormaps.append(covert2colorMap(amplitude_envelope))


        ## original pattern amplitude map

        # v = images[gap_num * 12 + j]
        # v2 = images[gap_num * 12 + (j + 1) % 4]
        v = filtered_original_diff_patterns_list[gap_num][j]
        v2 = filtered_original_diff_patterns_list[gap_num][(j + 1) % 4]

        amplitude_envelope_natural = np.sqrt(v**2 + v2**2)

        amplitude_natural_maps.append(amplitude_envelope_natural)
        amplitude_natural_colormaps.append(covert2colorMap(amplitude_envelope_natural))




    amplitude_maps_list.append(amplitude_maps)
    amplitude_original_maps_list.append(amplitude_original_maps)

    amplitude_colormaps_list.append(amplitude_colormaps)
    amplitude_original_colormaps_list.append(amplitude_original_colormaps)

    amplitude_natural_maps_list.append(amplitude_natural_maps)
    amplitude_natural_colormaps_list.append(amplitude_natural_colormaps)


#############################################################################


################## amplitude map image show ###################################

for gap, amplitude_maps in enumerate(amplitude_maps_list):

    for idx, amplitude_map in enumerate(amplitude_maps):

        cv2.imshow("amplitude_map", amplitude_map)
        cv2.waitKey(0)

        save_img(f"{base_path}/amplitude_map/", amplitude_map, gap, idx)


    amplitude_sum_image = np.sum(amplitude_maps, axis=0)


    #amplitude_sum_image = amplitude_sum_image/np.max(amplitude_sum_image)

    amplitude_sum_image_color = covert2colorMap(amplitude_sum_image)

    cv2.imshow("amplitude_sum_image_color", amplitude_sum_image_color)
    cv2.waitKey(0)

    save_img(f"{base_path}/amplitude_sum_image_color/", amplitude_sum_image_color, gap, f"amplitude_sum_image_color_gap_{gap}")

for gap, amplitude_original_maps in enumerate(amplitude_original_maps_list):

    for idx, amplitude_original_map in enumerate(amplitude_original_maps):

        cv2.imshow("amplitude_original_map", amplitude_original_map)
        cv2.waitKey(0)

        save_img(f"{base_path}/amplitude_original_map/", amplitude_original_map, gap, idx)


    amplitude_original_sum_image = np.sum(amplitude_original_maps, axis=0)


    #amplitude_sum_image = amplitude_sum_image/np.max(amplitude_sum_image)

    amplitude_original_sum_image_color = covert2colorMap(amplitude_original_sum_image)

    cv2.imshow("amplitude_original_sum_image_color", amplitude_original_sum_image_color)
    cv2.waitKey(0)

    save_img(f"{base_path}/amplitude_original_sum_image_color/", amplitude_original_sum_image_color, gap, f"amplitude_original_sum_image_color_gap_{gap}")


for gap, amplitude_natural_maps in enumerate(amplitude_natural_maps_list):

    # for amplitude_original_map in amplitude_original_maps:

    #     cv2.imshow("amplitude_original_map", amplitude_original_map)
    #     cv2.waitKey(0)


    amplitude_natural_sum_image = np.sum(amplitude_natural_maps, axis=0)


    #amplitude_sum_image = amplitude_sum_image/np.max(amplitude_sum_image)

    amplitude_natural_sum_image_color = covert2colorMap(amplitude_natural_sum_image)

    cv2.imshow("amplitude_natural_sum_image_color", amplitude_natural_sum_image_color)
    cv2.waitKey(0)
    
    save_img(f"{base_path}/amplitude_natural_sum_image_color/", amplitude_natural_sum_image_color, gap, f"amplitude_natural_sum_image_color_gap_{gap}")
    
#breakpoint()

### color map ###

for gap, amplitude_colormaps in enumerate(amplitude_colormaps_list):

    for idx, amplitude_colormap in enumerate(amplitude_colormaps):

        cv2.imshow("amplitude_colormap", amplitude_colormap)
        cv2.imshow("amplitude_colormap_uint8", (amplitude_colormap * 255).astype(np.uint8))
        cv2.imshow("amplitude_colormap_uint8", cv2.convertScaleAbs(amplitude_colormap, alpha=255.0))
        cv2.waitKey(0)

        save_img(f"{base_path}/amplitude_colormap/", amplitude_colormap, gap, idx)

for gap, amplitude_original_colormaps in enumerate(amplitude_original_colormaps_list):

    for idx, amplitude_original_colormap in enumerate(amplitude_original_colormaps):

        cv2.imshow("amplitude_original_colormap", amplitude_original_colormap)
        cv2.waitKey(0)

        save_img(f"{base_path}/amplitude_original_colormap/", amplitude_original_colormap, gap, idx)

################################################################################