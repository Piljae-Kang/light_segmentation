import corner_detection as cd
import glob
import cv2
import numpy as np


if __name__ == "__main__":


    # 체커보드의 차원 정의
    CHECKERBOARD = (4,5) # 체커보드 행과 열당 내부 코너 수
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # 각 체커보드 이미지에 대한 3D 점 벡터를 저장할 벡터 생성
    objpoints = []
    # 각 체커보드 이미지에 대한 2D 점 벡터를 저장할 벡터 생성
    imgpoints = [] 
    # 3D 점의 세계 좌표 정의
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    projector_imgs = []

    for i in range(1, 23):

        print(f"{i}th pose image")
        horizontal_images_path = glob.glob(f"/home/piljae/Dataset/Hubitz/calibration/patternSetChoose/{i}/horizontal/*")
        vertical_images_path = glob.glob(f"/home/piljae/Dataset/Hubitz/calibration/patternSetChoose/{i}/vertical/*")

        H, checkerboard_img, overlaped_img = cd.compute_homography(horizontal_images_path, vertical_images_path)
        
        # cv2.imshow("checkerboard_img", checkerboard_img)
        # cv2.waitKey(0)

        checkerboard_gray = cv2.cvtColor(checkerboard_img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(checkerboard_gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret == True:
            objpoints.append(objp)
            projector_imgs.append(overlaped_img.copy())
            corners2 = cv2.cornerSubPix(checkerboard_gray, corners, (11, 11), (-1, -1), criteria)

            img = cv2.drawChessboardCorners(checkerboard_gray, CHECKERBOARD, corners2, ret)

            # if ret:
            #     for i, corner in enumerate(corners2):
            #         # 코너에 번호 텍스트 추가 (0, 1, 2, ...)
            #         cv2.putText(img, str(i), (int(corner[0][0]), int(corner[0][1])), 
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    
            # cv2.imshow('Corners with Numbers', img)
            # cv2.waitKey(0)

            corners_camera = np.array(corners2)
            corners_camera = corners_camera.astype(np.float32)
            corners_camera = corners_camera[:, :, ::-1]


            # for corner in corners_camera:
            #     cv2.circle(checkerboard_gray, (int(round(corner[0][1])), int(round(corner[0][0]))), 2, (0, 0, 1), -1)

            # cv2.imshow("checkerboard_gray", checkerboard_gray)
            # cv2.waitKey(0)

            corners_proj = cv2.perspectiveTransform(corners_camera.reshape(-1, 1, 2), H)

            # for corner in corners_proj:
            #     cv2.circle(overlaped_img, (int(round(corner[0][1])), int(round(corner[0][0]))), 2, (0, 0, 1), -1)
            
            # cv2.imshow("use_homography", overlaped_img)
            # cv2.waitKey(0)



            imgpoints.append(corners_proj)
    
    proj_image = cv2.imread("/home/piljae/Dropbox/hubitz/pattern/hubitz_projector/my_pattern_images/horizontal/pattern0_0.png")

    proj_image_gray = cv2.cvtColor(proj_image, cv2.COLOR_BGR2GRAY)

    if img is not None:  # 이미지가 성공적으로 읽어왔을 때만 처리
        h, w = img.shape[:2]  # 마지막 이미지의 높이와 너비 가져옴
    # 알려진 3D 점(objpoints) 값과 감지된 코너의 해당 픽셀 좌표(imgpoints) 전달, 카메라 캘리브레이션 수행
    if proj_image_gray is not None:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, proj_image_gray.shape[::-1], None, None)

    print("Camera matrix : \n")
    print(mtx)
    print("rvecs : \n")
    print(rvecs)
    print("tvecs : \n")
    print(tvecs)

    # reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error

        # breakpoint()

        # for corner in imgpoints2:
        #     cv2.circle(projector_imgs[i], (int(round(corner[0][1])), int(round(corner[0][0]))), 2, (0, 0, 1), -1)
        
        # cv2.imshow(f"projector_imgs[{i}]", projector_imgs[i])
        # cv2.waitKey(0)


    print("Total reprojection error: ", total_error / len(objpoints))