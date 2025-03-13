import cv2
import glob
import os
import numpy as np
import open3d as o3d


def hubitz_local2camera(points, hubitz_camera_matrix):

    # points : N x 3
    # points_homogenous : N x 4
    points_homogenous = np.hstack( (points, np.ones((points.shape[0], 1))) )

    # points_cc_homogenous : 4 x N
    points_cc_homogenous = hubitz_camera_matrix @ points_homogenous.T

    # point_cc : N x 3
    point_cc = points_cc_homogenous.T[:, :3]

    return point_cc

def hubitz_camera2pixel(points, K):

    fx, fy, cx, cy = K[:]

    depths = points[:, 2]

    points /= depths.reshape(-1, 1)

    points_pixel = np.zeros_like(points)
    points_pixel[:, 0] = points[:, 0] * fx + cx - 0.5
    points_pixel[:, 1] = points[:, 1] * fy + cy - 0.5
    points_pixel[:, 2] = depths

    return points_pixel, depths

def projected_img(points_pixel, resolution):


    img = np.zeros(resolution, dtype=np.float32)
    points_pixel = np.round(points_pixel).astype(int)

    rows = points_pixel[:, 1]
    cols = points_pixel[:, 0]
    
    h, w = resolution
    mask = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)

    valid_rows = rows[mask]
    valid_cols = cols[mask]
    img[valid_rows, valid_cols] = 1.0 
    # for pixel in points_pixel:

    #     if (pixel[1] >=0 and pixel[1] < h) and (pixel[0] >=0 and pixel[0] < w):
    #         print(h)
    #         print(pixel[1])
    #         print(pixel[0])
    #         img[pixel[1], pixel[0]] = 1

    return img


path = "/media/piljae/X31/experiment_result/fast_adaptive_pattern_experiment/gold_crown/1/gaussian/alpha_20"
path = "/media/piljae/X31/experiment_result/original_hubitz_pattern/gold_crown/4"

hubitz_pcd_path = f"{path}/com0_error.ply"
image = cv2.imread(f"{path}/com0.png")

amplitude_map = cv2.imread(f"{path}/amplitude_map.png")

image[:, :, 1] = image[:, :, 0]
image[:, :, 2] = image[:, :, 0]



hubitz_K = [3126.057861, 3128.052979, 158.818115, 202.218994]
hK = np.identity(3)
hK [ 0, 0]  = hubitz_K[0]
hK [ 1, 1]  = hubitz_K[1]
hK [ 0, 2]  = hubitz_K[2]
hK [ 1, 2]  = hubitz_K[3]

# hubitz_camera_matrix = np.array([[-0.982974, 0.0358881, 0.180207, -4.68843],
#                                 [0.0491517, 0.996358, 0.0696835, 1.8196],
#                                 [-0.17705, 0.0773545, -0.981157, 119.762],
#                                 [ 0, 0, 0, 1]])

hubitz_camera_matrix = np.array([[ -0.989284, 0.021003, -0.144489, 0.000000],
  [0.004941, 0.993848, 0.110638, 0.000000],
    [0.145924, 0.108739, -0.983302, 0.000000],
    [1.628670, 1.503018, 118.362206, 1.000000]]).T


hubitz_point_cloud = o3d.io.read_point_cloud(hubitz_pcd_path)
hubitz_points = np.asarray(hubitz_point_cloud.points)

#hubitz_points /= 1.2

hubitz_points_cc = hubitz_local2camera(hubitz_points.copy(), hubitz_camera_matrix)

hubitz_points_pixel, hubitz_depths = hubitz_camera2pixel(hubitz_points_cc.copy(), hubitz_K)

hubitz_projected_img = projected_img(hubitz_points_pixel, (480, 400))

index = hubitz_projected_img == 1.0
image[index] = (0, 0, 255)

cv2.imshow("hubitz_projected_img", hubitz_projected_img)
cv2.imshow("image", image)

# rotated_img = cv2.rotate(hubitz_projected_img, cv2.ROTATE_90_CLOCKWISE)
# cv2.imshow("hubitz_projected_imgt", rotated_img)
cv2.waitKey(0)


#####################################################################################################

### amplitude map

amplitude_map = cv2.rotate(amplitude_map, cv2.ROTATE_90_CLOCKWISE)
cv2.imshow("amplitude_map", amplitude_map)
cv2.waitKey(0)

h, w = amplitude_map.shape[:2]

colors = []
for i in range(len(hubitz_points_pixel)):
    u, v, _ = hubitz_points_pixel[i]
    u_rounded = int(round(u))
    v_rounded = int(round(v))

    # 이미지 내부에 있는지 확인
    if 0 <= u_rounded < w and 0 <= v_rounded < h:
        # amplitude_map이 컬러인지(3채널), 그레이스케일인지(1채널)에 따라 처리
        # 예: 3채널 BGR -> R,G,B로 사용
        if len(amplitude_map.shape) == 3 and amplitude_map.shape[2] == 3:
            b, g, r = amplitude_map[v_rounded, u_rounded]

            colors.append([r/255.0, g/255.0, b/255.0])
        else:
            amp = amplitude_map[v_rounded, u_rounded]
            colors.append([amp/255.0, amp/255.0, amp/255.0])
    else:

        colors.append([0.0, 0.0, 0.0])

# 6. 컬러가 포함된 PointCloud 생성
colored_pcd = o3d.geometry.PointCloud()
colored_pcd.points = o3d.utility.Vector3dVector(hubitz_points)
colored_pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

# 7. .ply 파일로 저장
o3d.io.write_point_cloud(f"{path}/amplitude_colored_error.ply", colored_pcd)
print(f"Saved amplitude-colored point cloud to: amplitude_colored.ply")
