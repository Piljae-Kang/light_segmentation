import new_corner_detection as cd2
import glob
import cv2
import numpy as np
import os

if __name__ == "__main__":


    # 체커보드의 차원 정의
    CHECKERBOARD = (8,11) # 체커보드 행과 열당 내부 코너 수
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # 각 체커보드 이미지에 대한 3D 점 벡터를 저장할 벡터 생성
    objpoints = []
    # 각 체커보드 이미지에 대한 2D 점 벡터를 저장할 벡터 생성
    imgpoints = [] 
    # 3D 점의 세계 좌표 정의
    imgpoints_camera = []

    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    projector_imgs = []
    checkerboard_imgs = []

    for i in range(0, 10):

        print(f"{i}th pose image")
        horizontal_images_path = glob.glob(f"//home/piljae/Dataset/Hubitz/new_calibration/calibration/{i}/horizontal/*")
        vertical_images_path = glob.glob(f"//home/piljae/Dataset/Hubitz/new_calibration/calibration/{i}/vertical/*")
        mask_folder_path = f"/home/piljae/Dataset/Hubitz/new_calibration/calibration/{i}/mask"
        if not os.path.exists(mask_folder_path):
            os.makedirs(mask_folder_path)


        H, overlaped_img = cd2.compute_homography(horizontal_images_path, vertical_images_path, mask_folder_path)

        if i == 2:
            my_H = H

        checkerboard_img = cv2.imread(glob.glob(f"//home/piljae/Dataset/Hubitz/new_calibration/calibration/{i}/checkerboard/*")[0])
        checkerboard_gray = cv2.cvtColor(checkerboard_img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(checkerboard_gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret == True:
            objpoints.append(objp)
            projector_imgs.append(overlaped_img.copy())
            checkerboard_imgs.append(checkerboard_img)
            corners2 = cv2.cornerSubPix(checkerboard_gray, corners, (11, 11), (-1, -1), criteria)

            img = cv2.drawChessboardCorners(checkerboard_gray, CHECKERBOARD, corners2, ret)

            # if ret:
            #     for i, corner in enumerate(corners2):
            #         # 코너에 번호 텍스트 추가 (0, 1, 2, ...)
            #         cv2.putText(img, str(i), (int(corner[0][0]), int(corner[0][1])), 
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    
            # cv2.imshow('Corners with Numbers', img)
            # cv2.waitKey(0)
            imgpoints_camera.append(corners2)
            corners_camera = np.array(corners2)
            corners_camera = corners_camera.astype(np.float32)
            corners_camera = corners_camera[:, :, ::-1]

            # for corner in corners_camera:
            #     cv2.circle(checkerboard_gray, (int(round(corner[0][1])), int(round(corner[0][0]))), 2, (0, 0, 1), -1)

            # cv2.imshow(f"checkerboard_gray{i}", checkerboard_gray)
            # cv2.waitKey(0)

            corners_proj = cv2.perspectiveTransform(corners_camera.reshape(-1, 1, 2), H)

            # for corner in corners_proj:
            #     cv2.circle(overlaped_img, (int(round(corner[0][1])), int(round(corner[0][0]))), 2, (0, 0, 1), -1)
            
            # cv2.imshow("use_homography", overlaped_img)
            # cv2.waitKey(0)



            imgpoints.append(corners_proj)
    
    proj_image = cv2.imread("/home/piljae/Dropbox/hubitz/pattern/hubitz_projector/my_pattern_images/horizontal/pattern0_0.png")

    proj_image_gray = cv2.cvtColor(proj_image, cv2.COLOR_BGR2GRAY)

    
    print("Homography matrix : \n")
    print(my_H)

    # 알려진 3D 점(objpoints) 값과 감지된 코너의 해당 픽셀 좌표(imgpoints) 전달, 카메라 캘리브레이션 수행
    if proj_image_gray is not None:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, proj_image_gray.shape[::-1], None, None)

    print("Projector matrix : \n")
    print(mtx)
    print("Projector rvecs : \n")
    print(rvecs)
    print("Projector tvecs : \n")
    print(tvecs)

    # reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error

    print("Total reprojection error: ", total_error / len(objpoints))


    ############## 이건 카메라 calibration ###############

    ret_c, mtx_c, dist_c, rvecs_c, tvecs_c = cv2.calibrateCamera(objpoints, imgpoints_camera, checkerboard_gray.shape[::-1], None, None)


    print("Camera matrix : \n")
    print(mtx_c)
    print("Camera rvecs : \n")
    print(rvecs_c)
    print("Camera tvecs : \n")
    print(tvecs_c)

    # reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs_c[i], tvecs_c[i], mtx_c, dist_c)
        error = cv2.norm(imgpoints_camera[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error

        # for corner in imgpoints2:
        #     cv2.circle(checkerboard_imgs[i], (int(round(corner[0][0])), int(round(corner[0][1]))), 2, (0, 0, 1), -1)

        # cv2.imshow(f"checkerboard_imgs[{i}] : {error}", checkerboard_imgs[i])
        # cv2.waitKey(0)
        


    print("Total reprojection error: ", total_error / len(objpoints))


    ############ 검증

    def calculate_homography(K_proj, K_cam, rvecs, rvecs_c, tvecs, tvecs_c):
        # Rotation vector를 Rotation matrix로 변환
        R_proj, _ = cv2.Rodrigues(rvecs)
        R_cam, _ = cv2.Rodrigues(rvecs_c)

        # Homogeneous 변환 행렬 구성 (4x4)
        RT_proj = np.eye(4)
        RT_proj[:3, :3] = R_proj
        RT_proj[:3, 3] = tvecs.flatten()

        RT_cam = np.eye(4)
        RT_cam[:3, :3] = R_cam
        RT_cam[:3, 3] = tvecs_c.flatten()

        # Intrinsic 행렬을 homogeneous로 확장
        K_proj_h = np.eye(4)
        K_proj_h[:3, :3] = K_proj

        K_cam_h = np.eye(4)
        K_cam_h[:3, :3] = K_cam

        # Homography 계산 (4x4 계산 후 상단 3x3 추출)
        H_full = K_proj_h @ RT_proj @ np.linalg.inv(RT_cam) @ np.linalg.inv(K_cam_h)
        H = H_full[:3, :3]  # 상단 3x3 추출

        print(RT_proj @ np.linalg.inv(RT_cam))
        print(H)

        breakpoint()

        return H

    # Homography 비교 함수
    def compare_homographies(my_H, calculated_H):
        # Frobenius norm 계산
        frobenius_norm = np.linalg.norm(my_H - calculated_H, ord='fro')
        return frobenius_norm

    # 데이터 준비
    K_proj = mtx
    K_cam = mtx_c

    for i in range(8):
        # Homography 계산
        calculated_H = calculate_homography(K_proj, K_cam, rvecs[i], rvecs_c[i], tvecs[i], tvecs_c[i])

        # 비교
        norm = compare_homographies(my_H, calculated_H)

        # 출력
        print(f"Frame {i}: Frobenius norm = {norm}")

        # 기준값으로 검증 (예: 1e-3 이하로 설정)
        if norm < 1e-3:
            print(f"Frame {i}: Homographies match well!\n")
        else:
            print(f"Frame {i}: Homographies differ significantly!\n")