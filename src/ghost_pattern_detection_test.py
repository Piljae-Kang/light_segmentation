import cv2
import numpy as np
import glob
from scipy.ndimage import median_filter, gaussian_filter, gaussian_filter1d
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans


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

def covert2colorMap2(img):

    normalized_imgg = img/np.max(img)

    normalized_imgg[normalized_imgg >= 1.0] = 1.0

    gamma = 1
    normalized_imgg = np.power(normalized_imgg, gamma)

    normalized_imgg_uint8 = (normalized_imgg * 255).astype(np.uint8)

    colormap_imgg = cv2.applyColorMap(normalized_imgg_uint8, cv2.COLORMAP_JET)

    return colormap_imgg

def norm_gamma(img):

    normalized_imgg = img/np.mean(img)

    normalized_imgg[normalized_imgg >= 1.0] = 1.0

    gamma = 1
    normalized_imgg = np.power(normalized_imgg, gamma)

    return normalized_imgg


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
        alpha = 10
        syn_edge = sigmoid(alpha * filtered_edge_images[i])
        synthesize_images.append(syn_edge)


    for i in range(images_len):
        
        value1 = filtered_edge_images[i]
        value2 = filtered_edge_images[(i+1)%images_len]


        value1 = synthesize_images[i] - 0.5
        value2 = synthesize_images[(i+1)%images_len] - 0.5

        amplitude_envelope = np.sqrt(value1**2 + value2**2)

        amplitude_maps.append(amplitude_envelope)
        
        #original_envelope = np.sqrt(filtering(pattern_images[i], parameter=1.0, type=1)**2 + filtering(pattern_images[(i+2)%4], parameter=1.0, type=1)**2)
        original_envelope = np.maximum.reduce(pattern_images)
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
        # cv2.imshow("pattnern", pattern_images[i])
        # cv2.imshow("pattnern_", pattern_images[(i+1)%images_len])
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



def kmeans_spatial_color_clustering(
    img: np.ndarray,
    n_clusters: int = 8,
    divide_color_255: bool = True
):

    # 1. 이미지 형태 확인
    shape_len = len(img.shape)
    if shape_len == 2:
        # 그레이스케일
        H, W = img.shape
        is_color = False
    elif shape_len == 3 and img.shape[2] == 3:
        # 컬러
        H, W, _ = img.shape
        is_color = True
    else:
        raise ValueError("입력 이미지가 (H,W) 또는 (H,W,3) 형태가 아닙니다.")
    
    # 2. (H, W) → 좌표 행렬 x, y 만들기
    #    x: 0 ~ W-1, y: 0 ~ H-1
    #    이를 0~1 범위로 정규화하기 위해 (W-1), (H-1)로 나눔
    xs = np.repeat(np.arange(W), H).astype(np.float32)   # (H*W,)
    ys = np.tile(np.arange(H), W).astype(np.float32)     # (H*W,)

    xs = xs / max(W - 1, 1)  # 0~1 범위
    ys = ys / max(H - 1, 1)  # 0~1 범위

    pixel_scale = 0.2

    # 3. 픽셀 값(그레이 또는 RGB)을 펼쳐서 (H*W, 1 or 3)으로 만들기
    if is_color:
        # (H, W, 3) -> (H*W, 3)
        # 만약 img가 0~255 범위라면, 0~1 정규화(옵션)
        if divide_color_255:
            img_val = (img.reshape(-1, 3).astype(np.float32)) / 255.0
        else:
            img_val = img.reshape(-1, 3).astype(np.float32)
    else:

        img_val = img.reshape(-1, 1).astype(np.float32)

    for i in range(0, 1):

        # pixel_scale = np.power(2, i)

        if is_color:
            data = np.column_stack([
                img_val[:, 0],  # R
                img_val[:, 1],  # G
                img_val[:, 2],  # B
                xs * 1,
                ys * 1
            ])
        else:
            # (Gray, x, y) → 3차원
            data = np.column_stack([
                img_val[:, 0],  # Gray
                xs * 10,
                ys * 10
            ])

        # 5. K-Means 수행
        kmeans = KMeans(n_clusters=n_clusters, random_state=100)
        kmeans.fit(data)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # 6. 클러스터 중심(centers)에서 "색상/밝기" 부분만 추출
        #    - 그레이스케일: centers.shape = (n_clusters, 3) -> 첫 번째 요소가 밝기
        #    - 컬러: centers.shape = (n_clusters, 5) -> 처음 3개가 (R,G,B)
        if is_color:
            # (R, G, B)
            color_centers = centers[:, :3]  # shape = (n_clusters, 3)
        else:
            # (Gray)
            color_centers = centers[:, 0:1] # shape = (n_clusters, 1)

        # 7. 각 픽셀을 해당 클러스터 중심의 색(밝기)으로 치환
        segmented_data = color_centers[labels]

        # 8. (H, W, 3) 또는 (H, W) 로 재배열
        if is_color:
            segmented_img = segmented_data.reshape(H, W, 3)
        else:
            segmented_img = segmented_data.reshape(H, W)

        # 9. 값이 혹시 0~1 범위를 벗어났으면 clip (실제로는 거의 없겠지만 안전 차원)
        segmented_img = np.clip(segmented_img, 0.0, 1.0)

        cv2.imshow(f"segmented_img_{pixel_scale}", segmented_img)
        cv2.waitKey(0)

        # 클러스터마다 랜덤 색 생성 (0~1 범위)
        random_colors = np.random.rand(n_clusters, 3)

        # 각 픽셀 = 소속 클러스터의 랜덤 색
        segmented_data = random_colors[labels]
        segmented_data = np.clip(segmented_data, 0, 1)

        segmented_img = segmented_data.reshape(H, W, 3)


        cv2.imshow(f"random_segmented_img_{pixel_scale}", segmented_img)
        cv2.waitKey(0)


    # for i in range(n_clusters):
        
    #     cx_w = centers[i, 3] / 2  # 가중치 되돌리기
    #     cy_h = centers[i, 4] / 2

    #     cx_pixel = int(cx_w * (W - 1))
    #     cy_pixel = int(cy_h * (H - 1))

    #     cv2.circle(segmented_img, (cx_pixel, cy_pixel), radius=3, color=(0, 0, 255), thickness=-1)
        
    # 4) 각 클러스터별로 이미지를 따로 만들어 보기


    for i in range(n_clusters):
        # (H*W,) 형태의 boolean mask
        cluster_mask = (labels == i)

        # 새 배열을 만들기 (H*W, 3) float
        # 원본 RGB를 복사한 뒤, i번 클러스터가 아닌 곳은 (0, 0, 0) 등으로 처리
        cluster_pixels = np.zeros_like(img_val)
        cluster_pixels[cluster_mask] = img_val[cluster_mask]

        # (H, W, 3)로 리쉐이프
        cluster_img_rgb = cluster_pixels.reshape(H, W, 3)

        # 시각화를 위해 0~1 범위가 아니라면, clip 및 타입 변환 등 필요
        # 여기서는 일단 float 값(실제로는 0~255 범위이지만 float형) 그대로 보여줄 수도 있고,
        # OpenCV로 보여주려면 uint8로 만들어주는 게 일반적
        cluster_img_rgb_clamped = np.clip(cluster_img_rgb, 0, 255).astype(np.uint8)

        # 원하는 방식대로 시각화 (matplotlib 예시)
        cv2.imshow("cluster_img_rgb_clamped", cluster_img_rgb_clamped)
        cv2.waitKey


    return segmented_img, labels, centers

    
    

material = "gold_crown"
date = "25_03_07"
pattern = "half_pattern"
experiment = "frame_experiment"
case_num = 3
# images_path = glob.glob(f"/media/piljae/X31/Dataset/Hubitz/devided_pattern/{material}/frame/*.png")
images_frames = glob.glob(f"/media/piljae/X31/Dataset/Hubitz/depth_compute/original_images/{material}/stream_data/{pattern}/{case_num}/*.png")

# images_frames = glob.glob(f"/media/piljae/X31/Dataset/Hubitz/depth_compute/original_images/gold_crown/half_pattern/5/original_pattern/*.png")

images_frames = sorted(images_frames, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

frames = 29

base_path = f"/media/piljae/X31/experiment_result/ghost_pattern_detection/{material}/{pattern}/{experiment}/{date}/case_num_{case_num}"

original_frame_data_path = f"/media/piljae/X31/Dataset/Hubitz/depth_compute/frame_data/{material}/original_syn_pattern/{case_num}"
half_frame_data_path = f"/media/piljae/X31/Dataset/Hubitz/depth_compute/frame_data/{material}/half_syn_pattern/{case_num}"

os.makedirs(original_frame_data_path, exist_ok=True)
os.makedirs(half_frame_data_path, exist_ok=True)


for frame_idx in range(int(len(images_frames)/frames)):

    # if frame_idx > 0:
    #     break
    
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
    original_pattern_high_45_ = images[8:12]
    original_pattern_low = images[12:16]
    original_pattern_mid = images[16:20]
    original_pattern_high_0 = images[20:24]
    orignal_rgb = images[24:28]
    black = [images[28]]


    # original_pattern_high_45 = images[0:4]
    # original_pattern_low = images[4:8]
    # original_pattern_mid = images[8:12]
    # original_pattern_high_0 = images[12:16]
    # orignal_rgb = images[16:20]
    # black = [images[20]]


    ori_45_syns, ori_45_syns_amps, ori_45_amps = synthesize_original_pattern(original_pattern_high_45)
    # ori_low_syns, ori_low_syns_amps, ori_low_amps = synthesize_original_pattern(original_pattern_low)
    # ori_mid_syns, ori_mid_syns_amps, ori_mid_amps = synthesize_original_pattern(original_pattern_mid)
    # ori_0_syns, ori_0_syns_amps, ori_0_amps = synthesize_original_pattern(original_pattern_high_0)
    
    # original_syn_frame_data = [element for lst in [ori_45_syns, ori_low_syns, ori_mid_syns, ori_0_syns, orignal_rgb, black] for element in lst]

    max_img = np.maximum.reduce(original_pattern_high_45)
    min_img = np.minimum.reduce(original_pattern_high_45)

    max_img = filtering(max_img, parameter=1.0, type=1)
    min_img = filtering(min_img, parameter=1.0, type=1)

    # max_img = filtering(max_img, parameter=3, type=0)
    # min_img = filtering(min_img, parameter=3, type=0)

    direct = max_img - min_img

    # direct = filtering(direct, parameter=3, type=0)
    direct = filtering(direct, parameter=1.0, type=1)

    indirect = 2 * min_img



    cropped_img = direct[141:246, 72:154]


    diiif = np.abs(max_img-direct)
    diiif /= np.max(diiif)



    diiif = np.power(diiif, 1/2.4)

    g_img = orignal_rgb[0]
    r_img = orignal_rgb[2]

    diff_gr = np.abs(g_img - r_img)
    diff_gr /= np.max(diff_gr)


    cv2.imshow("max_img", max_img)
    cv2.imshow("diff_gr", diff_gr)
    cv2.imshow("direct", direct)
    cv2.imshow("diiif", diiif)
    cv2.imshow("diiif_Color", covert2colorMap(diiif))
    cv2.imshow("diiif_Color2", covert2colorMap2(diiif))
    cv2.waitKey(0)
    

    for i in range(10):
        
        tmp = diiif.copy()

        ths = (0.5 + 0.01 * i)

        tmp [(tmp < ths )& (tmp > 0.45)] = 0

    # cv2.imshow("max_img", max_img)
    # cv2.imshow("direct", direct)
    # cv2.imshow("covert2colorMap_direct", covert2colorMap(direct))
    # cv2.imshow("covert2colorMap_max_img", covert2colorMap(max_img))
    # cv2.imshow("diff____2", covert2colorMap(np.abs(norm_gamma(direct)- norm_gamma(max_img))))
    # cv2.imshow("diff____3", np.abs(max_img-direct)/np.max(np.abs(max_img-direct)))
    # cv2.imshow("diiif", diiif)
    # cv2.imshow("diiif222", covert2colorMap2(max_img - np.abs(max_img-direct)/np.max(np.abs(max_img-direct))))
    # cv2.imshow("diff____1", covert2colorMap2(np.abs(norm_gamma(direct)- norm_gamma(max_img))))

        cv2.imshow(f"tmp_{ths}", tmp)
        cv2.waitKey(0)

    normalized_imgg_uint8 = (direct * 255).astype(np.uint8)

    colormap_imgg = cv2.applyColorMap(normalized_imgg_uint8, cv2.COLORMAP_JET)

    #direct_seg, labels, centers = kmeans_spatial_color_clustering(max_img * 255)
    direct_seg_rgb, labels, centers = kmeans_spatial_color_clustering(colormap_imgg)
    #colormap_imgg = covert2colorMap2(direct_seg)

    #cv2.imshow("colormap_imgg", colormap_imgg)

    cv2.imshow("direct_seg_rgb", direct_seg_rgb)
    #cv2.imshow("direct_seg", direct_seg)
    cv2.waitKey(0)


    

    hist_cv2 = cv2.calcHist([cropped_img], [0], None, [256], [0, 1])

    # 시각화
    plt.plot(hist_cv2, color='black')
    plt.title('Grayscale Histogram (0~1)')
    plt.xlabel('Intensity (0~1)')
    plt.ylabel('Frequency')
    plt.show()

    plt.hist(cropped_img.ravel(), bins=256, range=(0,1))
    plt.title('Histogram (0~1) - Bar')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    

    cv2.imshow("cropped_img", cropped_img)
    cv2.imshow("max", max_img)
    cv2.imshow("min", min_img)
    cv2.imshow("direct", direct)
    cv2.imshow("indirect", indirect)
    plt.show()
    cv2.waitKey(0)

    cv2.imshow("ori_45_syns0", ori_45_syns[0])
    cv2.imshow("ori_45_syns1", ori_45_syns[1])
    cv2.imshow("ori_45_syns2", ori_45_syns[2])
    cv2.imshow("ori_45_syns3", ori_45_syns[3])

    diff = np.abs(norm_gamma(ori_45_syns_amps[0])- norm_gamma(ori_45_amps[0]))
    # diff = np.abs(norm_gamma(max_img)- norm_gamma(direct))
    cv2.imshow("ori_45", ori_45_syns_amps[0])
    cv2.imshow("ori_45_syns_amps", covert2colorMap(ori_45_syns_amps[0]))
    cv2.imshow("ori_45_amps", covert2colorMap(ori_45_amps[0]))
    cv2.imshow("covert2colorMap_diff2", covert2colorMap2(diff))
    cv2.imshow("covert2colorMap_diff", covert2colorMap(diff))
    cv2.imshow("diff", diff)
    cv2.imshow("ori_45_amps - diff", covert2colorMap2(ori_45_amps[0] - diff))
    cv2.imshow("ori_45_syns_amps - diff", covert2colorMap2(ori_45_syns_amps[0] - diff))
    cv2.waitKey(0)


    
    print(f"{frame_idx} is Done!")