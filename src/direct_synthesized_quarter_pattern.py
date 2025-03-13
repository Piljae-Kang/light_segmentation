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

    normalized_imgg = img/np.mean(img)

    normalized_imgg[normalized_imgg >= 1.0] = 1.0

    gamma = 1
    normalized_imgg = np.power(normalized_imgg, gamma)

    normalized_imgg_uint8 = (normalized_imgg * 255).astype(np.uint8)

    colormap_imgg = cv2.applyColorMap(normalized_imgg_uint8, cv2.COLORMAP_JET)



    #plt.figure(figsize=(12, 5))
    
    # 원본 이미지 히스토그램
    # plt.figure(figsize=(12, 5))
    # plt.hist(img.ravel(), bins=50, range=(0,1))
    # plt.title('Original Image Pixel Distribution')
    # plt.xlabel('Pixel value')
    # plt.ylabel('Count')

    # plt.show()
    
    # # 정규화된 이미지 히스토그램
    # plt.figure(figsize=(12, 5))
    # plt.hist(normalized_imgg.ravel(), bins=50, range=(0,1))
    # plt.title('Normalized Image Pixel Distribution')
    # plt.xlabel('Pixel value')
    # plt.ylabel('Count')
    
    # plt.tight_layout()
    # plt.show()


    return colormap_imgg


def save_img(img, folder_path, image_name):

    gamma=0.5

    image_corrected = np.power(img, gamma)  # 감마 적용

    # 2. OpenCV는 0~255로 변환해야 제대로 저장됨
    image_8bit = cv2.normalize(image_corrected, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


    os.makedirs(folder_path, exist_ok=True)
    cv2.imwrite(f"{folder_path}/{image_name}.png", image_8bit)

def amplitude_thresholding(img):

    normalized_imgg = img/np.mean(img)

    normalized_imgg[normalized_imgg >= 1.0] = 1.0

    gamma = 0.5
    normalized_imgg = np.power(normalized_imgg, gamma)

    normalized_imgg_uint8 = (normalized_imgg * 255).astype(np.uint8)

    colormap_imgg = cv2.applyColorMap(normalized_imgg_uint8, cv2.COLORMAP_JET)



    mask = np.zeros_like(img)

    threshold = 0.5

    mask[normalized_imgg < threshold] = 1.0
    # cv2.imshow("colormap_imgg", colormap_imgg)
    # cv2.imshow("img", img)
    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)

    return mask

material = "gold_crown"
date = "25_03_07"
pattern = "quarter_pattern"
experiment = "frame_experiment"
case_num = 1
# images_path = glob.glob(f"/media/piljae/X31/Dataset/Hubitz/devided_pattern/{material}/frame/*.png")
images_frames = glob.glob(f"/media/piljae/X31/Dataset/Hubitz/depth_compute/original_images/{material}/stream_data/{pattern}/{case_num}/*.png")

images_frames = sorted(images_frames, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

frames = 37

base_path = f"/media/piljae/X31/experiment_result/ghost_pattern_detection/{material}/{pattern}/{experiment}/{date}/case_num_{case_num}"

gap_dict = {0:8, 1:12, 2:20}

check_frame_idx = -1


for frame_idx in range(int(len(images_frames)/frames)):

    # if frame_idx != check_frame_idx:
    #     continue
    
    images_path = images_frames[frame_idx* frames : (frame_idx+1) * frames]

    images = []
    for image_path in images_path:
        
        img = cv2.imread(image_path)
        img = img.astype(np.float32) / 255.0
        images.append(img)
        save_img(img, f"{base_path}/original_images/frame_{frame_idx}", image_path.split("/")[-1].split(".")[0])
        


    images = np.array(images)

    gap8_images = images[:12]
    gap12_images = images[12:24]
    gap20_images = images[24:36]

    syn_pattern_list = []
    filtered_edge_list = []

    syn_original_pattern_list = []
    filtered_original_edge_list = []

    filtered_original_diff_patterns_list = []

    pattern_img = images[32]
    save_img(pattern_img, f"{base_path}/original_pattern_frameimg", frame_idx)
    # print(frame_idx)
    # continue
    


    for gap_num in range(3):

        imgs = images[12*gap_num : 12*(gap_num+1)]

        gap8_images = gap12_images

        syn_patterns = []

        syn_original_patterns = []

        filtered_original_diff_patterns = []

        filtered_edges = []
        filterd_original_edges = []

        for i in range(8):

            ascending_max = np.maximum.reduce([imgs[(0 + i) % 8], imgs[(1 + i) % 8], imgs[(2 + i) % 8], imgs[(3 + i) % 8]])
            ascending_min = np.minimum.reduce([imgs[(0 + i) % 8], imgs[(1 + i) % 8], imgs[(2 + i) % 8], imgs[(3 + i) % 8], imgs[(4 + i) % 8], imgs[(5 + i) % 8], imgs[(6 + i) % 8], imgs[(7 + i) % 8]])
            descending_max = np.maximum.reduce([imgs[(4 + i) % 8], imgs[(5 + i) % 8], imgs[(6 + i) % 8], imgs[(7 + i) % 8]])
            descending_min = ascending_min

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

            #filtered_edge = filtering(gap8_pattern1_ascending - gap8_pattern1_descending, parameter=1.0, type=1)
            filtered_edge = filtering(ascending_L_d - descending_L_d, parameter=1.0, type=1)

            filtered_edges.append(filtered_edge)

            alpha = 5
            syn_edge = sigmoid(alpha * filtered_edge)

            # cv2.imshow("syn_edge", syn_edge)
            # cv2.waitKey(0)

            # cv2.imshow("syn_edge_original", syn_edge_original)
            # cv2.waitKey(0)

            syn_patterns.append(syn_edge)


        syn_pattern_list.append(syn_patterns)
        filtered_edge_list.append(filtered_edges)

        for j in range(4):

            filtered_pattern_original = filtering(imgs[8 + j] - imgs[8 + ((j+2)%4)], parameter=1.0, type=1)
            syn_pattern_original = sigmoid(alpha * filtered_pattern_original)


            syn_original_patterns.append(syn_pattern_original)
            filtered_original_diff_patterns.append(filtering(imgs[8 + j], parameter=1.0, type=1))
            filterd_original_edges.append(filtered_pattern_original)

        syn_original_pattern_list.append(syn_original_patterns)
        filtered_original_diff_patterns_list.append(filtered_original_diff_patterns)
        filtered_original_edge_list.append(filterd_original_edges)


    ####### image show pattern list ####################################

    for gap, syn_patterns in enumerate(syn_pattern_list):

        for idx, syn_pattern in enumerate(syn_patterns):

            # cv2.imshow("syn_pattern", syn_pattern)
            # cv2.waitKey(0)

            save_img(syn_pattern, f"{base_path}/syn_direct_pattern/gap_{gap_dict[gap]}/frame_{frame_idx}", f"{idx}")

    for gap, syn_original_patterns in enumerate(syn_original_pattern_list):

        for idx, syn_original_patterns in enumerate(syn_original_patterns):

            # cv2.imshow("syn_original_patterns", syn_original_patterns)
            # cv2.waitKey(0)

            save_img(syn_original_patterns, f"{base_path}/syn_original_patterns/gap_{gap_dict[gap]}/frame_{frame_idx}", f"{idx}")

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

            # value = syn_pattern_list[gap_num][i] - 0.5
            # value2 = syn_pattern_list[gap_num][(i + 2) % 8] - 0.5

            value = filtered_edge_list[gap_num][i]
            value2 = filtered_edge_list[gap_num][(i+2) % 8]

            amplitude_envelope = np.sqrt(value**2 + value2**2)

            #amplitude_envelope [amplitude_envelope < 0.01] = 0

            amplitude_maps.append(amplitude_envelope)
            amplitude_colormaps.append(covert2colorMap(amplitude_envelope))


        for j in range(4):

            ## syn original pattern amplitude map

            # value = syn_original_pattern_list[gap_num][j] - 0.5
            # value2 = syn_original_pattern_list[gap_num][(j + 1) % 4] - 0.5

            value = filtered_original_edge_list[gap_num][j]
            value2 = filtered_original_edge_list[gap_num][(j+1) % 4]

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

    amplitude_mask_list = []

    for gap, amplitude_maps in enumerate(amplitude_maps_list):

        amplitude_masks = []

        for idx, amplitude_map in enumerate(amplitude_maps):

            # cv2.imshow("amplitude_map", amplitude_map)
            # cv2.waitKey(0)

            amplitude_mask = amplitude_thresholding(amplitude_map)

            amplitude_masks.append(amplitude_mask)
            save_img(amplitude_map,f"{base_path}/amplitude_map/gap_gap_{gap_dict[gap]}/frame_{frame_idx}", f"{idx}")

        amplitude_mask_list.append(amplitude_masks)

        amplitude_sum_image = np.sum(amplitude_maps, axis=0)


        #amplitude_sum_image = amplitude_sum_image/np.max(amplitude_sum_image)

        amplitude_sum_image_color = covert2colorMap(amplitude_sum_image)

        # cv2.imshow("amplitude_sum_image_color", amplitude_sum_image_color)
        # cv2.waitKey(0)

        save_img(amplitude_sum_image_color,f"{base_path}/amplitude_sum_image_color/gap_{gap_dict[gap]}", f"{frame_idx}")

    for gap, amplitude_original_maps in enumerate(amplitude_original_maps_list):

        for idx, amplitude_original_map in enumerate(amplitude_original_maps):

            # cv2.imshow("amplitude_original_map", amplitude_original_map)
            # cv2.waitKey(0)

            mask = amplitude_thresholding(amplitude_original_map)
            save_img(amplitude_original_map, f"{base_path}/amplitude_original_map/frame_{frame_idx}", f"{idx}")


        amplitude_original_sum_image = np.sum(amplitude_original_maps, axis=0)


        #amplitude_sum_image = amplitude_sum_image/np.max(amplitude_sum_image)

        amplitude_original_sum_image_color = covert2colorMap(amplitude_original_sum_image)

        # cv2.imshow("amplitude_original_sum_image_color", amplitude_original_sum_image_color)
        # cv2.waitKey(0)

        save_img(amplitude_original_sum_image_color, f"{base_path}/amplitude_original_sum_image_color/gap_{gap_dict[gap]}", f"{frame_idx}")


    for gap, amplitude_natural_maps in enumerate(amplitude_natural_maps_list):

        # for amplitude_original_map in amplitude_original_maps:

        #     cv2.imshow("amplitude_original_map", amplitude_original_map)
        #     cv2.waitKey(0)


        amplitude_natural_sum_image = np.sum(amplitude_natural_maps, axis=0)


        #amplitude_sum_image = amplitude_sum_image/np.max(amplitude_sum_image)

        # cv2.imshow("amplitude_natural_sum_image", amplitude_natural_sum_image)

        amplitude_natural_sum_image_color = covert2colorMap(amplitude_natural_sum_image)

        # cv2.imshow("amplitude_natural_sum_image_color", amplitude_natural_sum_image_color)
        # cv2.waitKey(0)
        
        save_img(amplitude_natural_sum_image_color, f"{base_path}/amplitude_natural_sum_image_color/gap_{gap_dict[gap]}", f"{frame_idx}")
        
    #breakpoint()

    ### color map ###

    for gap, amplitude_colormaps in enumerate(amplitude_colormaps_list):

        for idx, amplitude_colormap in enumerate(amplitude_colormaps):

            # cv2.imshow("amplitude_colormap", amplitude_colormap)
            # cv2.imshow("amplitude_colormap_uint8", (amplitude_colormap * 255).astype(np.uint8))
            # cv2.imshow("amplitude_colormap_uint8", cv2.convertScaleAbs(amplitude_colormap, alpha=255.0))
            # cv2.waitKey(0)

            save_img(amplitude_colormap, f"{base_path}/amplitude_colormap/gap_{gap_dict[gap]}/frame_{frame_idx}", f"{idx}")

    for gap, amplitude_original_colormaps in enumerate(amplitude_original_colormaps_list):

        for idx, amplitude_original_colormap in enumerate(amplitude_original_colormaps):

            # cv2.imshow("amplitude_original_colormap", amplitude_original_colormap)
            # cv2.waitKey(0)

            save_img(amplitude_original_colormap, f"{base_path}/amplitude_original_colormap/gap_{gap_dict[gap]}/frame_{frame_idx}", f"{idx}")

    ################################################################################

    for gap_num in range(3):

        for i in range(8):

            if i % 2 == 1:
                continue

            amplitude_map = amplitude_maps_list[gap_num][i]

            mask_img = amplitude_mask_list[gap_num][i]

            syn_pattern = syn_pattern_list[gap_num][i]

            filtered_edge = filtered_edge_list[gap_num][i]
            adaptive_alpha_map = np.zeros_like(amplitude_map)
            adaptive_alpha_map[mask_img == 0] = 1/amplitude_map[mask_img == 0]


            alpha_scale = 2
            adaptive_syn_edge = sigmoid(alpha_scale * adaptive_alpha_map * filtered_edge)


            ####################################3

            idx = int(i/2)

            syn_original_pattern = syn_original_pattern_list[gap_num][idx]

            amplitude_map_org = amplitude_original_maps_list[gap_num][idx]
            filterined_original_edge = filtered_original_edge_list[gap_num][idx]
            #adaptive_alpha_map_original = np.zeros_like(amplitude_map_org)
            adaptive_alpha_map_original = 1/amplitude_map_org

            alpha_scale = 2
            adaptive_syn_edge_original = sigmoid(alpha_scale * adaptive_alpha_map_original * filterined_original_edge)

            #################################
            # original pattern image

            original_pattern_img = images[12 * gap + 8 + idx]

            originla_pattern_img_masking = original_pattern_img.copy()
            originla_pattern_img_masking[mask_img == 1.0] = 0.0 


            # if gap_num == 2:
            #     cv2.imshow("mask_img", mask_img)
            #     cv2.imshow("adaptive_syn_edge", adaptive_syn_edge)
            #     cv2.imshow("adaptive_syn_edge_original", adaptive_syn_edge_original)
            #     cv2.imshow("syn_pattern", syn_pattern)
            #     cv2.imshow("syn_original_pattern", syn_original_pattern)
            #     cv2.imshow("original_pattern_img", original_pattern_img)
            #     cv2.imshow("originla_pattern_img_masking", originla_pattern_img_masking)

            #     cv2.imwrite(f"adaptive_syn_edge_{idx}.png", (adaptive_syn_edge*255).astype(np.uint8))
            #     cv2.imwrite(f"adaptive_syn_edge_original{idx}.png", (adaptive_syn_edge_original*255).astype(np.uint8))
            #     cv2.imwrite(f"mask_img{idx}.png", (mask_img*255).astype(np.uint8))
            #     # cv2.imshow("filtered_edge", filtered_edge)
            #     cv2.waitKey(0)


    
    print(f"frame {frame_idx} is Done!")