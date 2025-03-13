import cv2
import numpy as np
import glob
import os
import open3d as o3d
from scipy.ndimage import median_filter, gaussian_filter, gaussian_filter1d
from scipy.optimize import curve_fit, fsolve

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def plane(xy, a, b, c):
    x, y = xy
    return a * x + b * y + c

def quadratic_surface(xy, a, b, c, d, e, f):
    x, y = xy
    return a * x**2 + b * y**2 + c * x * y + d * x + e * y + f

def fitting_plane(w_1):

    edge_img = np.ones_like(w_1)
    sign_change_vert = np.where((w_1[:-1, :] * w_1[1:, :]) < 0)
    edge_img[sign_change_vert] = 0

    cv2.imshow("w_1", w_1)
    cv2.imshow("edge_img", np.stack([edge_img] * 3, axis=2))
    cv2.waitKey(0)

    edge_img = np.ones_like(w_1)
    visited = np.zeros_like(w_1)

    subpixel_edges = []

    matplotlib.use("TkAgg") 

    for i, j in zip(sign_change_vert[0], sign_change_vert[1]):
        if visited[i, j] == 1:
            continue

        # 3x3 patch
        x_range = np.arange(max(0, j - 2), min(w_1.shape[1], j + 3))
        y_range = np.arange(max(0, i - 2), min(w_1.shape[0], i + 3))
        x_grid, y_grid = np.meshgrid(x_range, y_range)

        patch_values = w_1[y_range[:, None], x_range]
        if patch_values.size < 3:  # 충분한 데이터가 없으면 무시
            continue

        visited[y_range[:, None], j] = 1

        # plane fitting
        xy_data = (x_grid.ravel(), y_grid.ravel())
        z_data = patch_values.ravel()
        popt, _ = curve_fit(plane, xy_data, z_data)

        a, b, c = popt

        # z = 0일 때의 i 좌표 계산 (j 고정)
        subpixel_position_i = -(a * j + c) / b if b != 0 else i

        # if subpixel_position_i < y_range[0] or subpixel_position_i > y_range[-1]:
        #     continue

        subpixel_edges.append((subpixel_position_i, j))

        if 0 <= int(subpixel_position_i) < edge_img.shape[0]:
            edge_img[int(subpixel_position_i), j] = 0


        # # 3D 시각화
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')

        # # 원래 데이터 (3x3 패치)
        # ax.scatter(x_grid, y_grid, patch_values, c='blue', label='Patch Data', depthshade=True)

        # # 피팅된 평면
        # x_flat = np.linspace(x_range.min(), x_range.max(), 10)
        # y_flat = np.linspace(y_range.min(), y_range.max(), 10)
        # X_flat, Y_flat = np.meshgrid(x_flat, y_flat)
        # Z_flat = plane((X_flat, Y_flat), *popt)
        # ax.plot_surface(X_flat, Y_flat, Z_flat, alpha=0.6, color='orange', label='Fitted Plane')

        # # z = 0 평면
        # Z_zero = np.zeros_like(X_flat)
        # ax.plot_surface(X_flat, Y_flat, Z_zero, alpha=0.3, color='gray', label='z=0 Plane')

        # # 평면과 z=0이 만나는 직선
        # line_y = np.linspace(y_range.min(), y_range.max(), 50)
        # line_x = -(b * line_y + c) / a if a != 0 else np.full_like(line_y, x_range.mean())
        # line_z = np.zeros_like(line_y)
        # ax.plot(line_x, line_y, line_z, color='red', label='Intersection Line')

        # # j 고정된 서브픽셀 i 좌표
        # ax.scatter(j, subpixel_position_i, 0, color='green', s=100, label='Subpixel Point (i, j)', depthshade=True)

        # # 축 라벨 및 제목
        # ax.set_title(f"3D Visualization for Column {j}")
        # ax.set_xlabel("X (j)")
        # ax.set_ylabel("Y (i)")
        # ax.set_zlabel("Z (Pixel Value)")
        # ax.legend()
        # plt.show()
    
    return subpixel_edges, edge_img


def fitting_plane2(w_1):
    edge_img = np.ones_like(w_1)
    visited = np.zeros_like(w_1)

    subpixel_edges = []

    for i in range(1, w_1.shape[0] - 1):
        for j in range(1, w_1.shape[1] - 1):
            
            patch = w_1[i-1:i+2, j-1:j+2]
            
            patch_c = w_1[i-1, j-1:j+2]
            patch_c = np.append(patch_c, w_1[i+1, j-1:j+2])
            patch_c = np.append(patch_c, w_1[i, j])
            
            if np.min(np.abs(patch_c)) == np.abs(w_1[i, j]) and w_1[i-1, j] * w_1[i+1, j] <= 0: 
                x_range = np.arange(j - 1, j + 2)
                y_range = np.arange(i - 1, i + 2)
                x_grid, y_grid = np.meshgrid(x_range, y_range)

                xy_data = (x_grid.ravel(), y_grid.ravel())
                z_data = patch.ravel()

                popt, _ = curve_fit(plane, xy_data, z_data)
                a, b, c = popt

                subpixel_position_i = -(a * j + c) / b if b != 0 else i
                
                # if subpixel_position_i < y_range[0] or subpixel_position_i > y_range[-1]:
                #     continue

                if 0 <= int(subpixel_position_i) < edge_img.shape[0]:
                    subpixel_edges.append((subpixel_position_i, j))
                    edge_img[int(subpixel_position_i), j] = 0

    # cv2.imshow("w_1", w_1)
    # cv2.imshow("edge_img2", np.stack([edge_img] * 3, axis=2))
    # cv2.waitKey(0)

    return subpixel_edges, edge_img

def fitting_curve2(w_1):
    edge_img = np.ones_like(w_1)
    sign_change_vert = np.where((w_1[:-1, :] * w_1[1:, :]) < 0)
    edge_img[sign_change_vert] = 0

    edge_img = np.ones_like(w_1)
    visited = np.zeros_like(w_1)

    subpixel_edges = []

    for i in range(1, w_1.shape[0] - 1):
        for j in range(1, w_1.shape[1] - 1):

            patch = w_1[i-1:i+2, j-1:j+2]
            patch_c = w_1[i-1, j-1:j+2]
            patch_c = np.append(patch_c, w_1[i+1, j-1:j+2])
            patch_c = np.append(patch_c, w_1[i, j])

            if np.min(np.abs(patch_c)) == np.abs(w_1[i, j]) and w_1[i-1, j] * w_1[i+1, j] <= 0:
                x_range = np.arange(j - 1, j + 2)
                y_range = np.arange(i - 1, i + 2)
                x_grid, y_grid = np.meshgrid(x_range, y_range)

                xy_data = (x_grid.ravel(), y_grid.ravel())
                z_data = patch.ravel()

                try:
                    popt, _ = curve_fit(quadratic_surface, xy_data, z_data)
                except RuntimeError as e:
                    print(f"Curve fitting failed at (i={i}, j={j}): {e}")
                    continue

                a, b, c, d, e, f = popt

                def intersection(y):
                    return a * j**2 + b * y**2 + c * j * y + d * j + e * y + f

                subpixel_position_i = fsolve(intersection, x0=i)[0]

                
                
                if subpixel_position_i < y_range[0] or subpixel_position_i > y_range[-1]:
                    continue

                if 0 <= int(subpixel_position_i) < edge_img.shape[0]:
                    edge_img[int(subpixel_position_i), j] = 0
                    subpixel_edges.append((subpixel_position_i, j))
                    
                    
                    # # 3D 시각화
                    # fig = plt.figure(figsize=(10, 8))
                    # ax = fig.add_subplot(111, projection='3d')

                    # # 원래 데이터 (3x3 패치)
                    # ax.scatter(x_grid, y_grid, patch_values, c='blue', label='Patch Data', depthshade=True)

                    # # 피팅된 곡면
                    # x_flat = np.linspace(x_range.min(), x_range.max(), 50)
                    # y_flat = np.linspace(y_range.min(), y_range.max(), 50)
                    # X_flat, Y_flat = np.meshgrid(x_flat, y_flat)
                    # Z_flat = quadratic_surface((X_flat, Y_flat), *popt)
                    # ax.plot_surface(X_flat, Y_flat, Z_flat, alpha=0.6, color='orange', label='Fitted Surface')

                    # # z = 0 평면
                    # Z_zero = np.zeros_like(X_flat)
                    # ax.plot_surface(X_flat, Y_flat, Z_zero, alpha=0.3, color='gray', label='z=0 Plane')

                    # # 평면과 z=0이 만나는 직선
                    # intersection_line = []
                    # for y in y_flat:
                    #     x_val = fsolve(lambda x: quadratic_surface((x, y), *popt), x0=x_range.mean())[0]
                    #     intersection_line.append((x_val, y))

                    # intersection_line = np.array(intersection_line)
                    # ax.plot(intersection_line[:, 0], intersection_line[:, 1], np.zeros_like(intersection_line[:, 0]),
                    #         color='red', label='Intersection Line')

                    # # j 고정된 서브픽셀 i 좌표
                    # ax.scatter(j, subpixel_position_i, 0, color='green', s=100, label='Subpixel Point (i, j)', depthshade=True)

                    # # 축 라벨 및 제목
                    # ax.set_title(f"3D Visualization for Column {j}")
                    # ax.set_xlabel("X (j)")
                    # ax.set_ylabel("Y (i)")
                    # ax.set_zlabel("Z (Pixel Value)")
                    # ax.legend()
                    # plt.show()

    # cv2.imshow("w_1", w_1)
    # cv2.imshow("edge_img2_curve", np.stack([edge_img] * 3, axis=2))
    # cv2.waitKey(0)

    return subpixel_edges, edge_img

def fitting_curve(w_1):

    edge_img = np.ones_like(w_1)
    sign_change_vert = np.where((w_1[:-1, :] * w_1[1:, :]) < 0)
    edge_img[sign_change_vert] = 0

    # cv2.imshow("w_1", w_1)
    # cv2.imshow("edge_img", np.stack([edge_img] * 3, axis=2))
    # cv2.waitKey(0)

    edge_img = np.ones_like(w_1)
    visited = np.zeros_like(w_1)

    subpixel_edges = []

    matplotlib.use("TkAgg")

    for i, j in zip(sign_change_vert[0], sign_change_vert[1]):
        if visited[i, j] == 1:
            continue

        # 3x3 patch
        x_range = np.arange(max(0, j - 2), min(w_1.shape[1], j + 3))
        y_range = np.arange(max(0, i - 2), min(w_1.shape[0], i + 3))
        x_grid, y_grid = np.meshgrid(x_range, y_range)

        patch_values = w_1[y_range[:, None], x_range]
        if patch_values.size < 3:  # 충분한 데이터가 없으면 무시
            continue

        visited[y_range[:, None], j] = 1

        # 곡면 피팅
        xy_data = (x_grid.ravel(), y_grid.ravel())
        z_data = patch_values.ravel()
        try:
            popt, _ = curve_fit(quadratic_surface, xy_data, z_data)
        except RuntimeError as e:
            print(f"Curve fitting failed at (i={i}, j={j}): {e}")
            continue

        a, b, c, d, e, f = popt

        # z = 0일 때의 i 좌표 계산 (j 고정)
        def intersection(y):
            return a * j**2 + b * y**2 + c * j * y + d * j + e * y + f

        subpixel_position_i = fsolve(intersection, x0=i)[0]

        subpixel_edges.append((subpixel_position_i, j))

        if 0 <= int(subpixel_position_i) < edge_img.shape[0]:
            edge_img[int(subpixel_position_i), j] = 0

        # # 3D 시각화
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')

        # # 원래 데이터 (3x3 패치)
        # ax.scatter(x_grid, y_grid, patch_values, c='blue', label='Patch Data', depthshade=True)

        # # 피팅된 곡면
        # x_flat = np.linspace(x_range.min(), x_range.max(), 50)
        # y_flat = np.linspace(y_range.min(), y_range.max(), 50)
        # X_flat, Y_flat = np.meshgrid(x_flat, y_flat)
        # Z_flat = quadratic_surface((X_flat, Y_flat), *popt)
        # ax.plot_surface(X_flat, Y_flat, Z_flat, alpha=0.6, color='orange', label='Fitted Surface')

        # # z = 0 평면
        # Z_zero = np.zeros_like(X_flat)
        # ax.plot_surface(X_flat, Y_flat, Z_zero, alpha=0.3, color='gray', label='z=0 Plane')

        # # 평면과 z=0이 만나는 직선
        # intersection_line = []
        # for y in y_flat:
        #     x_val = fsolve(lambda x: quadratic_surface((x, y), *popt), x0=x_range.mean())[0]
        #     intersection_line.append((x_val, y))

        # intersection_line = np.array(intersection_line)
        # ax.plot(intersection_line[:, 0], intersection_line[:, 1], np.zeros_like(intersection_line[:, 0]),
        #         color='red', label='Intersection Line')

        # # j 고정된 서브픽셀 i 좌표
        # ax.scatter(j, subpixel_position_i, 0, color='green', s=100, label='Subpixel Point (i, j)', depthshade=True)

        # # 축 라벨 및 제목
        # ax.set_title(f"3D Visualization for Column {j}")
        # ax.set_xlabel("X (j)")
        # ax.set_ylabel("Y (i)")
        # ax.set_zlabel("Z (Pixel Value)")
        # ax.legend()
        # plt.show()

    return subpixel_edges, edge_img

def find_3D_point(K_cam, K_proj, RT_cam, RT_proj, point, projector_h_line_index):
    """
    Calculate the 3D coordinates of a point based on the camera and projector parameters.

    Parameters:
        K_cam (np.ndarray): Camera intrinsic matrix (3x3).
        K_proj (np.ndarray): Projector intrinsic matrix (3x3).
        RT_cam (np.ndarray): Camera extrinsic matrix (4x4).
        RT_proj (np.ndarray): Projector extrinsic matrix (4x4).
        point (np.ndarray): Pixel coordinates in the camera image (x, y).
        projector_h_line_index (int): Horizontal line index of the projector (0 to 719).

    Returns:
        np.ndarray: 3D coordinates of the point in world space (X, Y, Z).
    """

    # Extract camera parameters
    R_cam = RT_cam[:3, :3]
    t_cam = RT_cam[:3, 3]
    cam_origin_world = -np.linalg.inv(R_cam) @ t_cam  # Camera origin in world coordinates

    # Extract projector parameters
    R_proj = RT_proj[:3, :3]
    t_proj = RT_proj[:3, 3]
    proj_origin_world = -np.linalg.inv(R_proj) @ t_proj  # Projector origin in world coordinates

    # Back-project the camera pixel to a ray in camera space
    pixel_homogeneous = np.array([point[1], point[0], 1.0])  # Homogeneous coordinates
    ray_cam = np.linalg.inv(K_cam) @ pixel_homogeneous

    # Convert the ray to world space
    ray_world = np.linalg.inv(R_cam) @ ray_cam

    # Define the plane for the projector's horizontal line
    line_proj_homogeneous = np.array([0, projector_h_line_index, 1.0])
    point_on_line_proj = np.linalg.inv(K_proj) @ line_proj_homogeneous  # Projector space
    #point_on_line_world = np.linalg.inv(R_proj) @ (point_on_line_proj - t_proj)  # World space
    point_on_line_world = R_proj.T @ point_on_line_proj + t_proj

    # The plane normal is aligned with the projector's y-axis
    normal_proj = np.linalg.inv(R_proj) @ np.array([0, 1, 0])

    # Plane equation: normal_proj \dot (X - point_on_line_world) = 0

    # Ray-plane intersection formula
    numerator = np.dot(normal_proj, (point_on_line_world - cam_origin_world))
    denominator = np.dot(normal_proj, ray_world)

    if np.isclose(denominator, 0):
        raise ValueError("Ray is parallel to the plane, no intersection found.")

    t = numerator / denominator

    # Calculate the intersection point
    intersection_world = cam_origin_world + t * ray_world
    
    breakpoint()
    
    points_h = np.hstack((intersection_world, 1))  # (4,)
    cam_points = (RT_proj @ points_h)  # (4,) -> 카메라 좌표계 변환
    # z > 0인 포인트만 남기기 (카메라 앞쪽)

    # 카메라 투영
    projected = (K_proj @ cam_points[:3].T).T  # (N, 3)
    projected[0] /= projected[2]  # x / z
    projected[1] /= projected[2]  # y / z
    
    breakpoint()

    return intersection_world



material = "metal_shaft"
images_path = glob.glob(f"/media/piljae/X31/Dataset/Hubitz/depth_compute/original_images/{material}/1/*")
images_path.sort()

K_proj = np.load("camera_projector_parameter/K_proj.npy")
K_cam = np.load("camera_projector_parameter/K_cam.npy")
RT_proj = np.load("camera_projector_parameter/RT_proj.npy")
RT_cam = np.load("camera_projector_parameter/RT_cam.npy")

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

pattern_info_path = glob.glob("/media/piljae/X31/Dataset/Hubitz/pattern_info/*")
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

# cv2.imshow("horizontal_pattern_decimal_2D_RGB", horizontal_pattern_decimal_2D_RGB)
# cv2.imshow("binary_images_RGB", binary_images_RGB)
# cv2.imshow("images", images[0])
# cv2.waitKey(0)


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
    
    file = open("point_cloud.xyz", "w")

    

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
        
        median_filtered_w1 = filtering(w_1, parameter=3, type=0)
        print("!!!")
        gaussian_filtered_w1 = filtering(w_1, parameter=1.0, type=1)
        print("gaussian!!")

        subpixel_edges_1_plane, egde_img_1_plane = fitting_plane2(w_1)
        
        subpixel_edges_1_curve, egde_img_1_curve = fitting_curve2(w_1)
        
        subpixel_edges_1_plane_median, egde_img_1_plane_median = fitting_plane2(median_filtered_w1)
        subpixel_edges_1_curve_plane_median, egde_img_1_curve_plane_median = fitting_curve2(median_filtered_w1)
        
        subpixel_edges_1_plane_gaussian, egde_img_1_plane_gaussian = fitting_plane2(gaussian_filtered_w1)
        subpixel_edges_1_curve_gaussian, egde_img_1_curve_gaussian = fitting_curve2(gaussian_filtered_w1)

        breakpoint()

        cv2.imshow("median_filtered_w1", median_filtered_w1)
        cv2.imshow("gaussian_filtered_w1", gaussian_filtered_w1)       
        cv2.imshow("egde_img_1_plane", egde_img_1_plane)
        cv2.imshow("egde_img_1_curve", egde_img_1_curve)
        cv2.imshow("egde_img_1_plane_median", egde_img_1_plane_median)
        cv2.imshow("egde_img_1_curve_plane_median", egde_img_1_curve_plane_median)
        cv2.imshow("egde_img_1_plane_gaussian", egde_img_1_plane_gaussian)
        cv2.imshow("egde_img_1_curve_gaussian", egde_img_1_curve_gaussian)
        cv2.waitKey(0)
        
        #egde_img_1 = egde_img_1_plane

        count = 0

        check_edge = np.ones_like(w_1)
        for i, point in enumerate(subpixel_edges_1_curve):
            
            range = np.arange(max(0, int(point[0]) - 2), min(w_1.shape[0], int(point[0]) + 3))
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
                continue

            # cv2.circle(egde_img_1, (int(point[1]), int(point[0])),1, (0, 0, 255), -1)
            # cv2.imshow("egde_img_1", egde_img_1)
            # for index in transition_indices:
            #     horizontal_pattern_decimal_2D_RGB[index, :] = (255, 255, 255)

            #     cv2.imshow("horizontal_pattern_decimal_2D_RGB_", horizontal_pattern_decimal_2D_RGB)
                # cv2.waitKey(0)
            # print("Transition Indices:", transition_indices)

            if len(transition_indices) != 0:
                count += 1
                check_edge[int(point[0]), int(point[1])] = 0

            else:
                print("here")
                print(transition_indices)
                continue

            depth_point = find_3D_point(K_cam, K_proj, RT_cam, RT_proj, point, transition_indices[0])
            
            file.write(f"{depth_point[0]} {depth_point[1]} {depth_point[2]}\n")
            
            print(i)
            
            # if i == 1000:
            #     break

        cv2.imshow("check_edge", check_edge)
        cv2.waitKey(0)
        print(f"count : {count}")
        file.close()
        break