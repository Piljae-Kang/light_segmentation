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

    return mask

def direct_synthesize_half_pattern(half_pattern_images):

    images_len = len(half_pattern_images)

    direct_images = []
    filtered_edge_images = []
    amplitude_maps = []
    synthesize_images = []
    direct_synthesize_images = []

    for i in range(images_len):

        img1 = half_pattern_images[i]
        img2 = half_pattern_images[(i+1)%images_len]
        img3 = half_pattern_images[(i+2)%images_len]
        img4 = half_pattern_images[(i+3)%images_len]

        L_plus = np.maximum(img1, img2)
        #L_plus2 = np.maximum(img3, img4)
        #L_plus = img1 + img2
        L_minus = np.minimum.reduce([img1, img2, img3, img4])

        L_d = L_plus - 2 * L_minus
        L_g = L_minus

        # cv2.imshow("img1", img1)
        # cv2.imshow("img2", img2)

        # cv2.imshow("L_plus", L_plus)
        # cv2.imshow("L_minus", L_minus)
        # cv2.imshow("L_d", L_d)
        # cv2.imshow("L_g", L_g)
        # cv2.waitKey(0)

        direct_images.append(L_d)

    for i in range(images_len):

        img = direct_images[i]
        img_inv = direct_images[(i+2)%images_len]

        edge_img = img - img_inv
        filtered_edge = filtering(edge_img, parameter=1.0, type=1)
        filtered_edge_images.append(filtered_edge)

    for i in range(images_len):
        
        value1 = filtered_edge_images[i]
        value2 = filtered_edge_images[(i+1)%images_len]

        amplitude_envelope = np.sqrt(value1**2 + value2**2)
        amplitude_maps.append(amplitude_envelope)

        alpha = 10
        syn_edge = sigmoid(alpha * filtered_edge_images[i])
        synthesize_images.append(syn_edge)

        #mask = amplitude_envelope < 0.01

        adaptive_alpha = np.zeros_like(amplitude_envelope)

        adaptive_alpha[amplitude_envelope > 0.001] = 0.1/amplitude_envelope[amplitude_envelope > 0.001]

        direct_syn_edge = sigmoid(alpha * adaptive_alpha * filtered_edge_images[i])

        direct_synthesize_images.append(direct_syn_edge)

        # cv2.imshow("direct_syn_edge", direct_syn_edge)
        # cv2.imshow("syn_edge", syn_edge)
        # cv2.imshow("covert2colorMap", covert2colorMap(amplitude_envelope))
        # cv2.waitKey(0)


    return direct_synthesize_images, amplitude_maps


def synthesize_original_pattern(pattern_images):

    images_len = len(pattern_images)

    filtered_edge_images = []
    amplitude_maps = []
    original_amplitude_mpas = []
    synthesize_images = []
    aasf_synthesize_images = []

    for i in range(images_len):

        img = pattern_images[i]
        img_inv = pattern_images[(i+2)%images_len]

        edge_img = img - img_inv
        filtered_edge = filtering(edge_img, parameter=1.0, type=1)
        filtered_edge_images.append(filtered_edge)

    for i in range(images_len):
        
        value1 = filtered_edge_images[i]
        value2 = filtered_edge_images[(i+1)%images_len]

        amplitude_envelope = np.sqrt(value1**2 + value2**2)

        amplitude_maps.append(amplitude_envelope)

        original_envelope = np.sqrt(pattern_images[i]**2 + pattern_images[(i+2)%images_len]**2)
        original_amplitude_mpas.append(original_envelope)

        alpha = 10
        syn_edge = sigmoid(alpha * filtered_edge_images[i])
        synthesize_images.append(syn_edge)

        #mask = amplitude_envelope < 0.01

        adaptive_alpha = np.zeros_like(amplitude_envelope)

        adaptive_alpha[amplitude_envelope > 0.001] = 0.1/amplitude_envelope[amplitude_envelope > 0.001]

        aasf_syn_edge = sigmoid(alpha * adaptive_alpha * filtered_edge_images[i])

        aasf_synthesize_images.append(aasf_syn_edge)

        # cv2.imshow("aasf_syn_edge", aasf_syn_edge)
        # cv2.imshow("syn_edge", syn_edge)
        # cv2.imshow("covert2colorMap_aasf", covert2colorMap(amplitude_envelope))
        # cv2.imshow("covert2colorMap_original", covert2colorMap(original_envelope))
        # cv2.waitKey(0)

    return aasf_synthesize_images, amplitude_maps, original_amplitude_mpas

def image_show_in_list(images):
    
    for img in images:
        cv2.imshow("img", img)
        cv2.waitKey(0)
        
def imwrite_in_list(images, folder_path, frame_idx, frames_num):
    
    for i, img in enumerate(images):
        
        num = frame_idx * frames_num + i
        
        cv2.imwrite(f"{folder_path}/{num}.png", (img * 255).astype(np.uint8))
    
    

material = "gold_crown"
date = "25_03_07"
pattern = "half_pattern"
experiment = "frame_experiment"
case_num = 6
# images_path = glob.glob(f"/media/piljae/X31/Dataset/Hubitz/devided_pattern/{material}/frame/*.png")

# 이건 그냥 고정된 1개 frame
images_frames = glob.glob(f"/media/piljae/X31/Dataset/Hubitz/depth_compute/original_images/{material}/{pattern}/{case_num}/*.png")

#images_frames = glob.glob(f"/media/piljae/X31/Dataset/Hubitz/depth_compute/original_images/{material}/stream_data/{pattern}/{case_num}/*.png")

images_frames = sorted(images_frames, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

frames = 29

base_path = f"/media/piljae/X31/experiment_result/ghost_pattern_detection/{material}/{pattern}/{experiment}/{date}/case_num_{case_num}"

original_frame_data_path = f"/media/piljae/X31/Dataset/Hubitz/depth_compute/frame_data/{material}/original_syn_pattern/{case_num}"
half_frame_data_path = f"/media/piljae/X31/Dataset/Hubitz/depth_compute/frame_data/{material}/half_syn_pattern/{case_num}"

os.makedirs(original_frame_data_path, exist_ok=True)
os.makedirs(half_frame_data_path, exist_ok=True)


for frame_idx in range(int(len(images_frames)/frames)):

    if frame_idx > 0:
        break
    
    images_path = images_frames[frame_idx* frames : (frame_idx+1) * frames]

    images = []
    for image_path in images_path:
        
        img = cv2.imread(image_path)
        img = img.astype(np.float32) / 255.0
        images.append(img)
        save_img(img, f"{base_path}/original_images/frame_{frame_idx}", image_path.split("/")[-1].split(".")[0])
        

    images = np.array(images)

    # data separation

    half_pattern_high_45 = images[:4]
    original_pattern_high_45 = images[4:8]
    original_pattern_low = images[8:12]
    original_pattern_mid = images[12:16]
    half_pattern_high_0 = images[16:20]
    original_pattern_high_0 = images[20:24]
    orignal_rgb = images[24:28]
    black = [images[28]]


    half_45_direct_syns, half_45_direct_syns_amps = direct_synthesize_half_pattern(half_pattern_high_45)
    ori_45_syns, ori_45_syns_amps, ori_45_amps = synthesize_original_pattern(original_pattern_high_45)
    ori_low_syns, ori_low_syns_amps, ori_low_amps = synthesize_original_pattern(original_pattern_low)
    ori_mid_syns, ori_mid_syns_amps, ori_mid_amps = synthesize_original_pattern(original_pattern_mid)
    half_0_direct_syns, half_0_direct_syns_amps = direct_synthesize_half_pattern(half_pattern_high_0)
    ori_0_syns, ori_0_syns_amps, ori_0_amps = synthesize_original_pattern(original_pattern_high_0)
    
    original_syn_frame_data = [element for lst in [ori_45_syns, ori_low_syns, ori_mid_syns, ori_0_syns, orignal_rgb, black] for element in lst]


    
    
    # half_0_direct_syns, half_0_direct_syns_amps = direct_synthesize_half_pattern(half_pattern_high_0)
    # ori_0_syns, ori_0_syns_amps, ori_0_amps = synthesize_original_pattern(original_pattern_high_0)
    
    half_syn_frame_data = []
    
    half_syn_frame_data.extend(half_45_direct_syns)
    half_syn_frame_data.extend(original_syn_frame_data[4:12])
    half_syn_frame_data.extend(half_0_direct_syns)
    half_syn_frame_data.extend(original_syn_frame_data[16:])

    
    
    # image_show_in_list(half_syn_frame_data)
    # image_show_in_list(original_syn_frame_data)
    
    
    # imwrite_in_list(original_syn_frame_data, original_frame_data_path, frame_idx, 21)
    # imwrite_in_list(half_syn_frame_data, half_frame_data_path, frame_idx, 21)



    ## 패턴 정리

    half_pattern_list = []

    half_pattern_list.extend(half_pattern_high_45)
    half_pattern_list.extend(original_pattern_low)
    half_pattern_list.extend(original_pattern_mid)
    half_pattern_list.extend(half_pattern_high_0)
    half_pattern_list.extend(images[24:])



    original_pattern_list = []

    original_pattern_list.extend(original_pattern_high_45)
    original_pattern_list.extend(original_pattern_low)
    original_pattern_list.extend(original_pattern_mid)
    original_pattern_list.extend(original_pattern_high_0)
    original_pattern_list.extend(images[24:])

    os.makedirs(f"/media/piljae/X31/Dataset/Hubitz/depth_compute/original_images/gold_crown/half_pattern/{case_num}/original_pattern", exist_ok=True)
    os.makedirs(f"/media/piljae/X31/Dataset/Hubitz/depth_compute/original_images/gold_crown/half_pattern/{case_num}/half_pattern", exist_ok=True)

    imwrite_in_list(original_pattern_list, f"/media/piljae/X31/Dataset/Hubitz/depth_compute/original_images/gold_crown/half_pattern/{case_num}/original_pattern", frame_idx, 21)
    imwrite_in_list(half_pattern_list, f"/media/piljae/X31/Dataset/Hubitz/depth_compute/original_images/gold_crown/half_pattern/{case_num}/half_pattern", frame_idx, 21)




    
    print(f"{frame_idx} is Done!")