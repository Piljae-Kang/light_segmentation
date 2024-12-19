import cv2
import numpy as np

def checkerboard_conv_method(image):

    # 이진화된 체커보드 이미지 로드


    # image_32bit = np.float32(image)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)

    # 2x2 컨볼루션 커널 정의
    kernel1 = np.array([[ -1, 1],
                        [ 1, -1]], dtype=np.float32)

    kernel2 = np.array([[ 1, -1],
                        [ -1, 1]], dtype=np.float32)

    conv_result1 = cv2.filter2D(image, -1, kernel1)
    conv_result2 = cv2.filter2D(image, -1, kernel2)
    corners = np.maximum(conv_result1, conv_result2)

    corners = np.where(corners > 300)

    # 이미지를 8비트로 변환 후, RGB로 변환
    image_uint8 = image.astype(np.uint8)
    image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
    # cv2.imshow('image', image_rgb)
    cv2.waitKey(0)

    # 변환된 이미지를 numpy 배열로 변환
    image_rgb = np.array(image_rgb)

    corners = np.array(corners)

    for corner in corners.T:
        image_rgb[corner[0], corner[1], 2] = 255

    cv2.imshow("corner",image_rgb)
    cv2.waitKey(0)
