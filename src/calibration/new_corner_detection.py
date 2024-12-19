import numpy as np
import cv2
import glob
from checkerboard_detector import checkerboard_conv_method
from structured_light import pattern_info_gen, structured_light_info_of_image, convert_7bit_to_rgb

import torch.nn.functional as F
import torch

def median_filter_4d(tensor, kernel_size=3,isnp=False, horizontal = False):

    if isnp:
       # np -> tensor
       tensor = torch.asarray(tensor)
       tensor = tensor.permute([0,3,1,2])

    tensor = tensor.float()

    padding = kernel_size // 2
    b, c, h, w = tensor.shape

    # Unfold the tensor to get sliding windows
    unfolded = F.unfold(tensor.view(b * c, 1, h, w), kernel_size, padding=padding)
    unfolded = unfolded.reshape(b, c, kernel_size , kernel_size, -1)
    if  horizontal:
        unfolded = unfolded [:,:,padding,:,:]
    else: 
        unfolded = unfolded [:,:,:,padding,:]
    # Apply median filtering
    median = unfolded.median(dim=2)[0]

    # Fold back to original shape
    median = median.view(b, c, h, w)

    #median = median.byte()

    if isnp:
       #tensor -> no
       median = median.permute([0,2,3,1])
       median = median.numpy()
    return median


def max_filter_4d(tensor, kernel_size=3,isnp=False, horizontal = False):

    if isnp:
       # np -> tensor
       tensor = torch.asarray(tensor)
       tensor = tensor.permute([0,3,1,2])

    tensor = tensor.float()

    padding = kernel_size // 2
    b, c, h, w = tensor.shape

    # Unfold the tensor to get sliding windows
    unfolded = F.unfold(tensor.view(b * c, 1, h, w), kernel_size, padding=padding)
    unfolded = unfolded.reshape(b, c, kernel_size , kernel_size, -1)
    if  horizontal:
        unfolded = unfolded [:,:,padding,:,:]
    else: 
        unfolded = unfolded [:,:,:,padding,:]
    # Apply median filtering
    median = unfolded.max(dim=2)[0]

    # Fold back to original shape
    median = median.view(b, c, h, w)

    #median = median.byte()

    if isnp:
       #tensor -> no
       median = median.permute([0,2,3,1])
       median = median.numpy()
    return median


def min_filter_4d(tensor, kernel_size=3,isnp=False, horizontal = False):

    if isnp:
       # np -> tensor
       tensor = torch.asarray(tensor)
       tensor = tensor.permute([0,3,1,2])

    tensor = tensor.float()

    padding = kernel_size // 2
    b, c, h, w = tensor.shape

    # Unfold the tensor to get sliding windows
    unfolded = F.unfold(tensor.view(b * c, 1, h, w), kernel_size, padding=padding)
    unfolded = unfolded.reshape(b, c, kernel_size , kernel_size, -1)
    if  horizontal:
        unfolded = unfolded [:,:,padding,:,:]
    else: 
        unfolded = unfolded [:,:,:,padding,:]
    # Apply median filtering
    median = unfolded.min(dim=2)[0]

    # Fold back to original shape
    median = median.view(b, c, h, w)

    #median = median.byte()

    if isnp:
       #tensor -> no
       median = median.permute([0,2,3,1])
       median = median.numpy()
    return median


def apply_gaussian_filter(image, kernel_size=3, sigma=1.0):
    # 이미지가 numpy 배열인 경우 torch tensor로 변환합니다.
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    image = image.float()  # 이미지를 float 형으로 변환

    padding = kernel_size // 2

    # 1D Gaussian kernel 생성
    gaussian_kernel = torch.tensor([np.exp(-(x - padding)**2 / (2 * sigma**2)) for x in range(kernel_size)], dtype=torch.float32)
    gaussian_kernel /= gaussian_kernel.sum()  # Normalize
    gaussian_kernel = gaussian_kernel.view(1, 1, -1, 1)  # 3x1 커널로 변환

    # 각 채널에 대해 Gaussian 필터 적용
    filtered_channels = []
    for i in range(image.shape[2]):  # 채널별로 처리
        channel = image[:, :, i].unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
        filtered_channel = F.conv2d(channel, gaussian_kernel, padding=(padding, 0))
        filtered_channels.append(filtered_channel.squeeze(0).squeeze(0))  # [1, 1, H, W] -> [H, W]

    # 채널들을 합쳐서 최종 이미지를 구성
    filtered_image = torch.stack(filtered_channels, dim=2)  # [H, W, C]

    # 결과 이미지를 numpy 배열로 변환하여 반환합니다.
    filtered_image = filtered_image.numpy()

    return filtered_image

def apply_horizontal_gaussian_filter(image, kernel_size=3, sigma=1.0):
    # 이미지가 numpy 배열인 경우 torch tensor로 변환합니다.
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    image = image.float()  # 이미지를 float 형으로 변환

    padding = kernel_size // 2

    # 1D Gaussian kernel 생성 (수평 방향)
    gaussian_kernel = torch.tensor([np.exp(-(x - padding)**2 / (2 * sigma**2)) for x in range(kernel_size)], dtype=torch.float32)
    gaussian_kernel /= gaussian_kernel.sum()  # Normalize
    gaussian_kernel = gaussian_kernel.view(1, 1, 1, -1)  # 1x3 커널로 변환

    # 각 채널에 대해 Gaussian 필터 적용
    filtered_channels = []
    for i in range(image.shape[2]):  # 채널별로 처리
        channel = image[:, :, i].unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
        filtered_channel = F.conv2d(channel, gaussian_kernel, padding=(0, padding))
        filtered_channels.append(filtered_channel.squeeze(0).squeeze(0))  # [1, 1, H, W] -> [H, W]

    # 채널들을 합쳐서 최종 이미지를 구성
    filtered_image = torch.stack(filtered_channels, dim=2)  # [H, W, C]

    # 결과 이미지를 numpy 배열로 변환하여 반환합니다.
    filtered_image = filtered_image.numpy()

    return filtered_image


def color_to_gray(image):

    return np.dot(image[...,:3], [0.299, 0.587, 0.114])

def find_change_points(vector):

    differences = np.diff(vector)

    nonzero_indices = np.nonzero(differences)[0]

    offset = nonzero_indices

    if len(nonzero_indices) == 1:
        return vector[nonzero_indices] , vector[nonzero_indices + 1], offset
    else:
        return None, None, None


def find_projector_pixel_point(h_structured_value_0, h_structured_value_1, horizontal_pattern_decimal, v_structured_value_0, v_structured_value_1, vertical_pattern_decimal):

    h_index = np.where((horizontal_pattern_decimal[:-1] == h_structured_value_0) & (horizontal_pattern_decimal[1:] == h_structured_value_1))[0]
    v_index = np.where((vertical_pattern_decimal[:-1] == v_structured_value_0) & (vertical_pattern_decimal[1:] == v_structured_value_1))[0]

    error_flag = 0

    if len(h_index) == 0 or len(v_index) == 0:
        error_flag = 1
        # breakpoint()
        
        return (0, 0), 1
        

    pixel_point = (h_index[0] + 1, v_index[0] + 1)

    return pixel_point, error_flag

from scipy.ndimage import convolve

def suppress_non_isolated_ones_numpy(image):
    # 3x3 커널 생성 (중심을 제외한 8방향의 합을 계산)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    # 주변 8방향의 값을 합산
    neighbor_sum = convolve(image, kernel, mode='constant', cval=0)
    
    # 조건에 맞는 위치는 0으로 만듦
    result_image = np.where((image == 1) & (neighbor_sum > 0), 0, image)
    
    return result_image

# # index 1 이미지만을 가지고 일단 checkerboard 만들거임
# horizontal_images_path = glob.glob("/home/piljae/Dataset/Hubitz/calibration/patternSetChoose/1/horizontal/*")
# vertical_images_path = glob.glob("/home/piljae/Dataset/Hubitz/calibration/patternSetChoose/1/vertical/*")

class ImageCropper:
    def __init__(self, image):
        self.image = image
        self.clone = image.copy()
        self.ref_point = []
        self.cropping = False

    def click_and_crop(self, event, x, y, flags, param):
        # 왼쪽 마우스 버튼이 눌렸을 때 시작 좌표 기록
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ref_point = [(x, y)]
            self.cropping = True

        # 마우스 이동 중일 때 (드래그 중일 때)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.cropping:
                # 현재 위치의 사각형 영역을 표시 (선택 영역 실시간 표시)
                image_copy = self.clone.copy()
                cv2.rectangle(image_copy, self.ref_point[0], (x, y), (0, 255, 0), 2)
                cv2.imshow("image", image_copy)

        # 왼쪽 마우스 버튼이 떼졌을 때 끝 좌표 기록
        elif event == cv2.EVENT_LBUTTONUP:
            self.ref_point.append((x, y))
            self.cropping = False

            # 최종 선택된 사각형 영역 표시
            cv2.rectangle(self.image, self.ref_point[0], self.ref_point[1], (0, 255, 0), 2)
            cv2.imshow("image", self.image)


def compute_homography(horizontal_images_path, vertical_images_path, mask_folder_path):

    horizontal_images_path = sorted(horizontal_images_path, key=lambda x: int(x.split('/')[-1][:-4]))
    vertical_images_path = sorted(vertical_images_path, key=lambda x: int(x.split('/')[-1][:-4]))
    mask_image_path = glob.glob(f"{mask_folder_path}/*")

    mask = None
    if len(mask_image_path) != 0:
        mask_int8 = cv2.imread(mask_image_path[0])
        mask = mask_int8 == 255

    h_img0s = []
    h_img1s = []
    v_img0s = []
    v_img1s = []
    diff_imgs = []
    corners_list = []

    for i in range(0, len(horizontal_images_path), 2):

        
        h_img_0 = cv2.imread(horizontal_images_path[i])
        h_img_1 = cv2.imread(horizontal_images_path[i+1])
        v_img_0 = cv2.imread(vertical_images_path[i])
        v_img_1 = cv2.imread(vertical_images_path[i+1])
        (h,w,c) = h_img_0.shape

        h_img_0 = h_img_0.astype(np.float32) / 255.0
        h_img_1 = h_img_1.astype(np.float32) / 255.0
        v_img_0 = v_img_0.astype(np.float32) / 255.0
        v_img_1 = v_img_1.astype(np.float32) / 255.0

        h_abs_diff_image = np.abs(h_img_0 - h_img_1)
        v_abs_diff_image = np.abs(v_img_0 - v_img_1)

        h_img_white = cv2.imread(horizontal_images_path[-2])
  
        h_img_white = h_img_white.astype(np.float32) / 255.0

        h_img_0_cp = h_img_0.copy()

        if mask is None:

            cropper = ImageCropper(h_img_0_cp)
            cv2.namedWindow("image")
            cv2.setMouseCallback("image", cropper.click_and_crop)

            while True:
                cv2.imshow("choose masking area", cropper.image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            
            mask = np.ones_like(h_img_white, dtype=bool)
            mask[cropper.ref_point[0][1]: cropper.ref_point[1][1], cropper.ref_point[0][0]: cropper.ref_point[1][0]] = False
            mask_uint8 = mask.astype(np.uint8) * 255
            cv2.imwrite(f"{mask_folder_path}/mask.png", mask_uint8)

        h_img_white_r = h_img_white
        h_img_white_r[mask] = 0

        # cv2.imshow("median_img_white.png", h_img_white)

        # mask = h_img_white < 0.99
        h_img_white_r[mask] = 1
        # cv2.imshow("h_img_white_r.png", h_img_white_r)

        my_img = np.ones_like(h_img_white)
        my_img[mask] = 0
        # cv2.imshow("my_img.png", my_img)
        # cv2.waitKey(0)

        h_abs_diff_image = np.abs(h_img_0 - h_img_1)
        h_abs_diff_image[mask] = 1
        v_abs_diff_image = np.abs(v_img_0 - v_img_1)
        v_abs_diff_image[mask] = 1

        # cv2.imshow("h_abs_diff_image.png", h_abs_diff_image)
        # cv2.imshow("v_abs_diff_image.png", v_abs_diff_image)
        # cv2.waitKey(0)


        line_img = min_filter_4d (h_abs_diff_image[None],horizontal=True,kernel_size=3, isnp=True)[0]
        line_mask = line_img < 0.5
        line = np.zeros_like(line_img)
        line[line_mask] = 1
        # cv2.imshow("line_img.png", line_img)
        # cv2.imshow("line.png", line)

        h_b_image = 1 - h_abs_diff_image
        v_b_image = 1 - v_abs_diff_image

        # cv2.imshow("h_b_image.png", h_b_image)
        # cv2.imshow("v_b_image.png", v_b_image)

        h_b_image = max_filter_4d (h_b_image[None],horizontal=True,kernel_size=7, isnp=True)[0]
        v_b_image = max_filter_4d (v_b_image[None],horizontal=False,kernel_size=7, isnp=True)[0]

        h_b_image = max_filter_4d (h_b_image[None],horizontal=True,kernel_size=7, isnp=True)[0]
        v_b_image = max_filter_4d (v_b_image[None],horizontal=False,kernel_size=7, isnp=True)[0]

        h_blurred_image = apply_gaussian_filter (h_b_image, kernel_size=5, sigma=0.2)
        v_blurred_image = apply_horizontal_gaussian_filter (v_b_image, kernel_size=5, sigma=0.2)
        # blurred_image = cv2.GaussianBlur(blurred_image, (3, 3), 0)
        # blurred_image = cv2.GaussianBlur(blurred_image, (3, 3), 0)

        h_blurred_image = max_filter_4d (h_blurred_image[None],horizontal=True,kernel_size=5, isnp=True)[0]
        v_blurred_image = max_filter_4d (v_blurred_image[None],horizontal=False,kernel_size=5, isnp=True)[0]

        h_line_mask = h_blurred_image > 0.9
        v_line_mask = v_blurred_image > 0.9

        h_line = np.zeros_like(line_img)
        v_line = np.zeros_like(line_img)
        h_line[h_line_mask] = 1
        v_line[v_line_mask] = 1

        # cv2.imshow("h_blurred_image.png", h_blurred_image)
        # cv2.imshow("h_line.png", h_line)
        # cv2.imshow("v_blurred_image.png", v_blurred_image)
        # cv2.imshow("v_line.png", v_line)


        corners_img = h_line * v_line

        corners_img[mask] = 0

        corners_img = np.mean(corners_img, axis=-1)

        corners_img = suppress_non_isolated_ones_numpy(corners_img)

        # cv2.imshow("corners_img.png", corners_img)
        # cv2.waitKey(0)


        corners = np.where(corners_img == 1)

        corners = np.vstack((corners[0], corners[1]))

        corners_list.append(corners)


        #######
        # 단순하게 그냥 diff 이미지 곱해서 구함

        # corner_img = h_abs_diff_image * v_abs_diff_image
        # inv_coner_img = 1 - corner_img

        # corners_s = corner_img == 0
        # corner_img = np.zeros_like(corner_img)
        # corner_img[corners] = 1


        # cv2.imshow("inv_coner_img.png", inv_coner_img)
        # cv2.imshow("corners_imgsss.png", corners_img)
        # cv2.waitKey(0)



        h_img0s.append( h_img_0)
        h_img1s.append( h_img_1)
        v_img0s.append( v_img_0)
        v_img1s.append( v_img_1)


    ## 각 레벨 마다의 코너 계산 완료

    horizontal_patterns_path = glob.glob("/home/piljae/Dropbox/hubitz/pattern/hubitz_projector/my_pattern_images/horizontal/*")
    vertical_patterns_path = glob.glob("/home/piljae/Dropbox/hubitz/pattern/hubitz_projector/my_pattern_images/vertical/*")

    horizontal_pattern_decimal, vertical_pattern_decimal, horizontal_pattern_string, vertical_pattern_string = pattern_info_gen(horizontal_patterns_path, vertical_patterns_path)

    horizontal_pattern_decimal_2D = np.tile(horizontal_pattern_decimal[:, np.newaxis], (1, 1280))
    horizontal_pattern_decimal_2D_RGB = convert_7bit_to_rgb(horizontal_pattern_decimal_2D)

    vertical_pattern_decimal_2D = np.tile(vertical_pattern_decimal, (720, 1))
    vertical_pattern_decimal_2D_RGB = convert_7bit_to_rgb(vertical_pattern_decimal_2D)

    # cv2.imshow("horizontal_pattern_decimal_2D_RGB", horizontal_pattern_decimal_2D_RGB)
    # cv2.imshow("vertical_pattern_decimal_2D_RGB", vertical_pattern_decimal_2D_RGB)
    # cv2.waitKey(0)


    # horizontal_images_path = glob.glob("/home/piljae/Dataset/Hubitz/calibration/patternSetChoose/1/horizontal/*")
    # vertical_images_path = glob.glob("/home/piljae/Dataset/Hubitz/calibration/patternSetChoose/1/vertical/*")

    h_result_binary, v_result_binary = structured_light_info_of_image(horizontal_images_path, vertical_images_path)

    camera_pixel_points = []
    projector_pixel_points = []
    for corner_pt in corners_list[0].T:

        
        horiozontal_gap_top = 1
        horiozontal_gap_bottom = 1
        if corner_pt[0] - horiozontal_gap_top < 0:
            horiozontal_gap_top = corner_pt[0]
        if corner_pt[0] + horiozontal_gap_bottom + 1 > h_result_binary.shape[0]:
            horiozontal_gap_bottom = h_result_binary.shape[0] - corner_pt[0] - 1
        
        horiozontal_patch_structured_light_info = h_result_binary[corner_pt[0] - horiozontal_gap_top : corner_pt[0] + horiozontal_gap_bottom + 1, corner_pt[1]]

        vertical_gap_left = 1
        vertical_gap_right = 1
        if corner_pt[1] - vertical_gap_left < 0:
            vertical_gap_left = corner_pt[1]
        if corner_pt[1] + vertical_gap_right + 1 > v_result_binary.shape[1]:
            vertical_gap_right = v_result_binary.shape[1] - corner_pt[1] - 1

        vertical_patch_structured_light_info = v_result_binary[corner_pt[0], corner_pt[1] - vertical_gap_left : corner_pt[1] + vertical_gap_right + 1]


        h_structured_value_0, h_structured_value_1, h_offset  = find_change_points(horiozontal_patch_structured_light_info)
        v_structured_value_0, v_structured_value_1, v_offset = find_change_points(vertical_patch_structured_light_info)

        if h_structured_value_0 is not None and v_structured_value_0 is not None:

            camera_pixel_point = (corner_pt[0] + h_offset[0], corner_pt[1] + v_offset[0])
            projector_pixel_point, error_flag = find_projector_pixel_point(h_structured_value_0, h_structured_value_1, horizontal_pattern_decimal, v_structured_value_0, v_structured_value_1, vertical_pattern_decimal)

            if error_flag == 0:
                camera_pixel_points.append(camera_pixel_point)
                projector_pixel_points.append(projector_pixel_point)

        # else:
        #     print(f"corner point : {corner_pt} is not corner")
        #     # breakpoint()


    camera_pixel_points = np.array(camera_pixel_points)
    projector_pixel_points = np.array(projector_pixel_points)

    h_img_0 = h_img0s[0].copy()
    v_img_0 = v_img0s[0].copy()

    for c_point, p_point in zip(camera_pixel_points, projector_pixel_points):

        cv2.circle(h_img_0, (c_point[1], c_point[0]), 2, (0, 0, 255), -1)
        cv2.circle(v_img_0, (c_point[1], c_point[0]), 2, (0, 0, 255), -1)
        cv2.circle(horizontal_pattern_decimal_2D_RGB, (p_point[1], p_point[0]), 2, (0, 0, 0), -1)
        cv2.circle(vertical_pattern_decimal_2D_RGB, (p_point[1], p_point[0]), 2, (0, 0, 0), -1)

    # cv2.imshow("h_img0s", h_img_0)
    # cv2.imshow("v_img0s", v_img_0)
    # cv2.imshow("horizontal_pattern_decimal_2D_RGB", horizontal_pattern_decimal_2D_RGB)
    # cv2.imshow("vertical_pattern_decimal_2D_RGB", vertical_pattern_decimal_2D_RGB)

    # cv2.waitKey(0)

    # 호모그래피 계산
    H, status = cv2.findHomography(camera_pixel_points, projector_pixel_points)


    # cv2.perspectiveTransform을 사용하기 전에 데이터 타입을 float32로 변환
    camera_pixel_points_float = camera_pixel_points.astype(np.float32)

    # 호모그래피 변환
    source_points_transformed = cv2.perspectiveTransform(camera_pixel_points_float.reshape(-1, 1, 2), H)

    # 재투영 오차 계산
    error = np.sqrt(np.sum((projector_pixel_points.reshape(-1, 1, 2) - source_points_transformed) ** 2, axis=2))
    mean_error = np.mean(error)

    print("Reprojection Error:", mean_error)

    # homography로 카메라에서 projector로 옮기기

    camera_image_width = 480
    camera_image_height = 400

    projector_image_width = 1280
    projector_image_height = 720

    camera_image = h_img0s[0].copy()
    # cv2.imwrite("camera_image.png", camera_image)
    # cv2.imshow("camera_image.png", camera_image)
    # cv2.waitKey(0)


    H_inv = np.linalg.inv(H)
    projector_image = np.zeros((projector_image_height, projector_image_width, 3))

    i_proj_coords, j_proj_coords = np.meshgrid(np.arange(projector_image_height), np.arange(projector_image_width), indexing='ij')

    projected_coords = np.stack([i_proj_coords + 0.5, j_proj_coords + 0.5, np.ones_like(i_proj_coords)], axis=-1)
    projected_coords_flat = projected_coords.reshape(-1, 3)
    camera_coords = np.dot(H_inv, projected_coords_flat.T).T

    camera_coords = camera_coords.reshape(projected_coords.shape)
    camera_coords /= np.expand_dims(camera_coords[:, :, 2], axis=-1)
    camera_coords = camera_coords[:, :, :2]

    camera_coords = np.round(camera_coords - 0.5).astype(int)

    false_mask = ~(mask[:, :, 0] & mask[:, :, 1] & mask[:, :, 2])

    # 유효한 좌표 계산
    valid_mask = (camera_coords[:, :, 0] >= 0) & (camera_coords[:, :, 0] < camera_image_height) & \
                (camera_coords[:, :, 1] >= 0) & (camera_coords[:, :, 1] < camera_image_width)
    
    # camera_coords가 false_mask의 범위(400, 480)을 벗어나지 않도록 조건 추가
    x_coords = camera_coords[:, :, 0].astype(int)
    y_coords = camera_coords[:, :, 1].astype(int)

    # x와 y 좌표가 false_mask의 유효한 인덱스 범위 안에 있는지 확인
    valid_x = (x_coords >= 0) & (x_coords < 400)
    valid_y = (y_coords >= 0) & (y_coords < 480)

    # x와 y가 모두 유효한 범위에 있는지 확인
    valid_coords = valid_x & valid_y

    # valid_coords가 True인 경우에만 false_mask 값을 가져옴
    false_mask_valid = np.zeros_like(valid_coords, dtype=bool)  # false_mask_valid를 초기화

    # 유효한 좌표들만 false_mask에서 인덱싱
    false_mask_valid[valid_coords] = false_mask[x_coords[valid_coords], y_coords[valid_coords]]

    # 최종적으로 valid_mask에 false_mask_valid 조건 추가
    valid_mask = valid_mask & false_mask_valid

    valid_proj_x = i_proj_coords[valid_mask]
    valid_proj_y = j_proj_coords[valid_mask]


    valid_cam_x = camera_coords[valid_mask][:, 0]
    valid_cam_y = camera_coords[valid_mask][:, 1]

    projector_image[valid_proj_x, valid_proj_y] = camera_image[valid_cam_x, valid_cam_y]

    # # mask를 2D로 변환: RGB 세 채널에서 하나라도 False인 부분을 False로 간주
    # false_mask = ~(mask[:, :, 0] & mask[:, :, 1] & mask[:, :, 2])

    # # false_mask에서 True인 부분에 해당하는 valid_proj 좌표에 camera_image를 할당
    # projector_image[valid_proj_x[false_mask.ravel()], valid_proj_y[false_mask.ravel()]] = camera_image[valid_cam_x[false_mask.ravel()], valid_cam_y[false_mask.ravel()]]

    # breakpoint()

    pattern_img = cv2.imread("/home/piljae/Dropbox/hubitz/pattern/hubitz_projector/my_pattern_images/horizontal/pattern0_0.png")

    pattern_img = pattern_img.astype(np.float32) / 255.0

    pattern_img[valid_proj_x, valid_proj_y] = projector_image[valid_proj_x, valid_proj_y]

    # cv2.imshow("camera_image.png", camera_image)
    # cv2.imshow("projector_image.png", projector_image)
    # cv2.imshow("pattern_img_overlap.png", pattern_img)
    # cv2.waitKey(0)

    return H, pattern_img