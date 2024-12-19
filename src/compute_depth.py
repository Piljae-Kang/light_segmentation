import cv2
import numpy as np
import glob
import os
import open3d as o3d
from scipy.ndimage import median_filter, gaussian_filter, gaussian_filter1d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def linear(x, m, c):
    return m * x + c

def calculate_depth(camera_points, projector_points, P_camera, P_projector):

    points_4d = cv2.triangulatePoints(P_camera, P_projector, camera_points, projector_points)
    
    points_3d = points_4d[:3] / points_4d[3]
    return points_3d.T

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

def binary_row_to_string(row):
    binary_string = ''.join(map(str, row))
    return binary_string

def binary_row_to_decimal(row):
    binary_string = ''.join(map(str, row))
    return int(binary_string, 2)

def pattern_info_gen(pattern_infos):

    pattern_infos = np.squeeze(pattern_infos, axis=2).T

    from collections import Counter

    pattern_string = np.apply_along_axis(binary_row_to_string, 1, pattern_infos)
    pattern_decimal = np.apply_along_axis(binary_row_to_decimal, 1, pattern_infos)

    pattern_counts = Counter(pattern_decimal)

    return pattern_decimal, pattern_string


def convert_7bit_to_rgb(image_7bit):

    # 비트 연산으로 RGB 채널 분리
    r = (image_7bit >> 5) & 0x07  # 상위 3비트
    g = (image_7bit >> 2) & 0x07  # 중간 3비트
    b = image_7bit & 0x03         # 하위 2비트

    # 7비트를 8비트 스케일로 확장 (0~7 → 0~255)
    r = (r * 255 // 7).astype(np.uint8)
    g = (g * 255 // 7).astype(np.uint8)
    b = (b * 255 // 3).astype(np.uint8)

    # RGB 배열 생성
    return np.stack([r, g, b], axis=-1)

def convert_12bit_to_rgb(image_12bit):
    """
    12비트 데이터를 8비트 RGB로 변환하는 함수
    """
    # 데이터 타입을 정수형으로 변환 (비트 연산을 위해 필요)
    image_12bit = image_12bit.astype(np.uint16)

    # 상위 4비트 → R
    r = (image_12bit >> 8) & 0x0F
    # 중간 4비트 → G
    g = (image_12bit >> 4) & 0x0F
    # 하위 4비트 → B
    b = image_12bit & 0x0F

    # 4비트를 8비트로 확장 (0~15 → 0~255)
    r = (r * 255 // 15).astype(np.uint8)
    g = (g * 255 // 15).astype(np.uint8)
    b = (b * 255 // 15).astype(np.uint8)

    # RGB 배열 생성
    return np.stack([r, g, b], axis=-1)

def find_edge_points(w):

    edges = np.array(np.where(w == 0)).T.astype(float)  # float 변환
    edges += 0.5
    return edges

def structured_light_info_of_image(images):
    
    h, w = images[0].shape[:2]
    binary_image = np.zeros((h,w,12), dtype=int)
    for i in range(3):
        
        image_t = images[4 * i : 4 * i + 4]
        
        image1 = image_t[0]
        image2 = image_t[1]
        image3 = image_t[2]
        image4 = image_t[3]
        
        index1 = image1 > image3
        index2 = image2 >= image4
        index3 = image3 > image1
        index4 = image4 >= image2 
        
        binary_image[index1, 4*i] = 1
        binary_image[index2, 4*i + 1] = 1
        binary_image[index3, 4*i + 2] = 1
        binary_image[index4, 4*i + 3] = 1
    
    decimal_image = np.dot(binary_image, 2 ** np.arange(12)[::-1])
    
    return decimal_image


import numpy as np

def decompose_homography(H, K_c, K_p):
    """
    Decompose Homography matrix into relative R (rotation) and T (translation).

    :param H: Homography matrix (3x3)
    :param K_c: Camera intrinsic matrix (3x3)
    :param K_p: Projector intrinsic matrix (3x3)
    :return: Rotation matrix R (3x3) and translation vector T (3x1)
    """
    # Normalize Homography with intrinsics
    H_normalized = np.linalg.inv(K_p) @ H @ K_c

    # Extract columns for R and T
    scale = np.linalg.norm(H_normalized[:, 0])  # Use the first column for scale
    R1 = H_normalized[:, 0] / scale
    R2 = H_normalized[:, 1] / scale
    T = H_normalized[:, 2] / scale

    # Ensure R is a proper rotation matrix
    R3 = np.cross(R1, R2)
    R = np.stack([R1, R2, R3], axis=1)

    # Orthogonalize R (optional, to handle numerical issues)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt  # Closest orthogonal matrix

    return R, T

def triangulate(camera_intrinsic, projector_intrinsic, R, T, camera_pixel):
    """
    Perform triangulation to compute the 3D point and depth.

    :param camera_intrinsic: Camera intrinsic matrix (3x3)
    :param projector_intrinsic: Projector intrinsic matrix (3x3)
    :param R: Relative rotation matrix (3x3)
    :param T: Relative translation vector (3x1)
    :param camera_pixel: Pixel coordinates in the camera (x, y)
    :return: 3D point (X, Y, Z) and depth Z
    """
    # Camera ray
    camera_pixel_h = np.array([*camera_pixel, 1])  # Homogeneous coordinates
    camera_ray = np.linalg.inv(camera_intrinsic) @ camera_pixel_h

    # Projector ray
    projector_ray = R @ camera_ray + T

    # Solve for intersection of rays using least squares
    A = np.stack([camera_ray, -projector_ray], axis=1)
    b = T
    lambdas = np.linalg.lstsq(A, b, rcond=None)[0]

    # Compute 3D point as midpoint of closest points
    point_camera = lambdas[0] * camera_ray
    point_projector = T + lambdas[1] * projector_ray
    point_3d = (point_camera + point_projector) / 2

    return point_3d, point_3d[2]  # Return 3D point and depth (Z)


material = "metal_shaft"
images_path = glob.glob(f"/media/piljae/X31/Dataset/Hubitz/depth_compute/original_images/{material}/1/*")
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

pattern_info_path = glob.glob(f"/media/piljae/X31/Dataset/Hubitz/pattern_info/*")
pattern_info_path.sort()

pattern_infos = []

for i in range(len(pattern_info_path) - 8):
    
    img = cv2.imread(pattern_info_path[i], cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    img = img[::-1]
    
    pattern_infos.append(img.astype(int))

pattern_infos = np.array(pattern_infos)

horizontal_pattern_decimal, vertical_pattern_decimal = pattern_info_gen(pattern_infos)
horizontal_pattern_decimal_2D = np.tile(horizontal_pattern_decimal[:, np.newaxis], (1, 1280))
horizontal_pattern_decimal_2D_RGB = convert_12bit_to_rgb(horizontal_pattern_decimal_2D)

binary_images = structured_light_info_of_image(images)
binary_images_RGB = convert_12bit_to_rgb(binary_images)

cv2.imshow("horizontal_pattern_decimal_2D_RGB", horizontal_pattern_decimal_2D_RGB)
cv2.imshow("binary_images_RGB", binary_images_RGB)
cv2.imshow("images", images[0])
cv2.waitKey(0)


for i in range(4):

    img_type = images[4 * i: 4 * i + 4]

    image1 = img_type[0]
    image2 = img_type[1]
    image3 = img_type[2]
    image4 = img_type[3]

    filtering_type = "nope"
    alpha = 100
    kernel_size = 3
    beta = 1
    gaussian_parameter = 1.0


    if filtering_type == "gaussian":
        # gaussian filtering
        w_1 = sigmoid(alpha * filtering(image1 - image3, parameter=gaussian_parameter, type=1), alpha=beta)
        w_2 = sigmoid(alpha * filtering(image2 - image4, parameter=gaussian_parameter, type=1), alpha=beta)
        w_3 = sigmoid(alpha * filtering(image3 - image1, parameter=gaussian_parameter, type=1), alpha=beta)
        w_4 = sigmoid(alpha * filtering(image4 - image2, parameter=gaussian_parameter, type=1), alpha=beta)

    else:
        w_1 = image1 - image3
        w_2 = image2 - image4
        w_3 = image3 - image1
        w_4 = image4 - image2

        #edge_points_w_1 = find_edge_points(w_1)

        edge_img = np.ones_like(w_1)
        sign_change_vert = np.where((w_1[:-1, :] * w_1[1:, :]) < 0)
        edge_img[sign_change_vert] = 0


        # edge_img = np.ones_like(w_1)
        # sign_change_vert = np.where((w_1[:-1, :] * w_1[1:, :]) < 0)
        # edge_img[sign_change_vert] = 0

        # cv2.imshow("w_1", w_1)
        # cv2.imshow("edge_img", np.stack([edge_img] * 3, axis=2))
        # cv2.waitKey(0)

        edge_img = np.ones_like(w_1)
        visited = np.zeros_like(w_1)
        subpixel_edges = []
        for i, j in zip(sign_change_vert[0], sign_change_vert[1]):

            if visited[i, j] == 1:
               continue

            y_range = np.arange(max(0, i - 1), min(w_1.shape[0], i + 2))
            values = w_1[y_range, j]


            visited[y_range, j] = 1

            x_data = np.arange(len(values))  # x 좌표 
            popt, _ = curve_fit(linear, x_data, values)


            m, c = popt
            subpixel_position = -c / m + y_range[0]  # 0 = mx + c에서 x 계산


            subpixel_edges.append((subpixel_position + 0.5, j + 0.5))


            if 0 <= int(subpixel_position + 0.5) < edge_img.shape[0]:  # 유효 범위 확인
                edge_img[int(subpixel_position + 0.5), int(j+0.5)] = 0

            # 그래프 그리기
            # plt.figure(figsize=(8, 6))
            # plt.plot(x_data + y_range[0], values, 'o', label="Data Points")
            # plt.plot(x_data + y_range[0], linear(x_data, *popt), '-', label=f"Fitted Line (y = {m:.2f}x + {c:.2f})")
            # plt.axhline(0, color='gray', linestyle='--', label="y=0")
            # plt.scatter(subpixel_position, 0, color='red', label="Subpixel Position")
            # plt.title(f"Subpixel Fitting for Column {j}")
            # plt.xlabel("Pixel Index")
            # plt.ylabel("Value")
            # plt.legend()
            # plt.grid(True)
            # plt.show()
            
            
        # cv2.imshow("edge_img2", edge_img)
        # cv2.waitKey(0)
        
        for point in subpixel_edges:
            
            range = np.arange(max(0, int(point[0]) - 2), min(w_1.shape[0], int(point[0]) + 2))
            #print(binary_images[range, int(point[1])])
            
            arr = binary_images[range, int(point[1])]
            arr_df = arr[:-1] - arr[1:]
            
            if len(np.where(arr_df == 0)[0]) == len(arr_df) -1:
                
                index = np.where(arr_df != 0)
                values_to_find = (arr[index[0]], arr[index[0] + 1])
            else:
                continue
                
            #print(values_to_find)
            #breakpoint()
            
            if values_to_find[0] in horizontal_pattern_decimal and values_to_find[1] in horizontal_pattern_decimal:
                
                transition_indices = np.where((horizontal_pattern_decimal[:-1] == values_to_find[0]) &
                                            (horizontal_pattern_decimal[1:] == values_to_find[1]))[0] + 1
            else:
                transition_indices = []

            # cv2.circle(edge_img, (int(point[1]), int(point[0])),1, (0, 0, 255), -1)
            # #cv2.imshow("edge_img_", edge_img)
            # for index in transition_indices:
            #     horizontal_pattern_decimal_2D_RGB[index, :] = (255, 255, 255)

                #cv2.imshow("horizontal_pattern_decimal_2D_RGB_", horizontal_pattern_decimal_2D_RGB)
                #cv2.waitKey(0)
            print("Transition Indices:", transition_indices)

            find_3D_point()