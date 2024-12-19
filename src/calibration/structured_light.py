import cv2
import numpy as np
import glob

### pattern info ###

def binary_row_to_string(row):
    binary_string = ''.join(map(str, row))
    return binary_string

def binary_row_to_decimal(row):
    binary_string = ''.join(map(str, row))
    return int(binary_string, 2)

def convert_7bit_to_rgb(image_7bit):
    # 빨강, 초록, 파랑 비트를 각각 추출 (3비트, 2비트, 2비트)
    r = (image_7bit >> 4) & 0x07  # 3비트 (상위 3비트)
    g = (image_7bit >> 2) & 0x03  # 2비트 (중간 2비트)
    b = image_7bit & 0x03         # 2비트 (하위 2비트)
    
    # 비트 값을 8비트로 확장
    r = np.uint8(np.round((r * 255) / 7))   # 3비트 -> 8비트
    g = np.uint8(np.round((g * 255) / 3))   # 2비트 -> 8비트
    b = np.uint8(np.round((b * 255) / 3))   # 2비트 -> 8비트
    
    # 3개의 채널로 합치기
    rgb_image = np.stack((r, g, b), axis=-1)
    
    return rgb_image


def pattern_info_gen(horizontal_patterns_path, vertical_patterns_path):

    horizontal_patterns_path = sorted(horizontal_patterns_path, key=lambda x: (int(x.split('/')[-1].split('_')[0][-1]), int(x.split('/')[-1].split('_')[1][0])))
    vertical_patterns_path = sorted(vertical_patterns_path, key=lambda x: (int(x.split('/')[-1].split('_')[0][-1]), int(x.split('/')[-1].split('_')[1][0])))


    horizontal_pattern_infos = []
    vertical_pattern_infos = []
    for i in range(0, len(horizontal_patterns_path), 2): # index 0 부터 시작 : 6 패턴까지

        print(vertical_patterns_path[i])
        horizontal_pattern_img = cv2.imread(horizontal_patterns_path[i])
        horizontal_pattern_info = horizontal_pattern_img[:, 0]
        horizontal_pattern_info = (np.mean(horizontal_pattern_info, axis=1, keepdims=True) / 255).astype(int)
        horizontal_pattern_infos.append(horizontal_pattern_info)
        
        vertical_pattern_img = cv2.imread(vertical_patterns_path[i])
        vertical_pattern_info = vertical_pattern_img[0, :]
        vertical_pattern_info = (np.mean(vertical_pattern_info, axis=1, keepdims=True) / 255).astype(int)
        vertical_pattern_infos.append(vertical_pattern_info)

    horizontal_pattern_infos = np.array(horizontal_pattern_infos)
    vertical_pattern_infos = np.array(vertical_pattern_infos)

    horizontal_pattern_infos = np.squeeze(horizontal_pattern_infos, axis=2).T
    vertical_pattern_infos = np.squeeze(vertical_pattern_infos, axis=2).T

    from collections import Counter

    horizontal_pattern_string = np.apply_along_axis(binary_row_to_string, 1, horizontal_pattern_infos)
    horizontal_pattern_decimal = np.apply_along_axis(binary_row_to_decimal, 1, horizontal_pattern_infos)

    vertical_pattern_string = np.apply_along_axis(binary_row_to_string, 1, vertical_pattern_infos)
    vertical_pattern_decimal = np.apply_along_axis(binary_row_to_decimal, 1, vertical_pattern_infos)

    horizontal_counts = Counter(horizontal_pattern_decimal)
    vertical_counts = Counter(vertical_pattern_decimal)


    return horizontal_pattern_decimal, vertical_pattern_decimal, horizontal_pattern_string, vertical_pattern_string


def structured_light_info_of_image(horizontal_images_path, vertical_images_path):

    
    horizontal_images_path = sorted(horizontal_images_path, key=lambda x: int(x.split('/')[-1].split('-')[-1][:-4]))
    vertical_images_path = sorted(vertical_images_path, key=lambda x: int(x.split('/')[-1].split('-')[-1][:-4]))

    horizontal_images = []
    vertical_images = []

    for i in range(len(horizontal_images_path)):
        
        horizontal_image = cv2.imread(horizontal_images_path[i], cv2.IMREAD_GRAYSCALE)
        horizontal_images.append(horizontal_image)

        vertical_image = cv2.imread(vertical_images_path[i], cv2.IMREAD_GRAYSCALE)
        vertical_images.append(vertical_image)



    # corner point가 어느 line에 위치하는 지 파악하기

    h_result_img = np.zeros((horizontal_image.shape[0], horizontal_image.shape[1], 7)).astype(int) # 지금 7개의 패턴 사용
    v_result_img = np.zeros((vertical_image.shape[0], vertical_image.shape[1], 7)).astype(int)

    for i in range(0, len(horizontal_images), 2):

        # breakpoint()
        h_img0 = horizontal_images[i]
        h_img1 = horizontal_images[i+1]
    
        v_img0 = vertical_images[i]
        v_img1 = vertical_images[i+1]


        h_index = h_img0 > h_img1 # 이게 1인 부분
        v_index = v_img0 > v_img1
        h_result_img[h_index, i//2] = 1
        v_result_img[v_index, i//2] = 1


    h_result_binary = np.sum(h_result_img * (2 ** np.arange(6, -1, -1)), axis=-1)
    v_result_binary = np.sum(v_result_img * (2 ** np.arange(6, -1, -1)), axis=-1)

    return h_result_binary, v_result_binary

######
    

if __name__ == "__main__":

    horizontal_patterns_path = glob.glob("/home/piljae/Dropbox/hubitz/pattern/hubitz_projector/my_pattern_images/horizontal/*")
    vertical_patterns_path = glob.glob("/home/piljae/Dropbox/hubitz/pattern/hubitz_projector/my_pattern_images/vertical/*")


    horizontal_pattern_decimal, vertical_pattern_decimal, horizontal_pattern_string, vertical_pattern_string = pattern_info_gen(horizontal_patterns_path, vertical_patterns_path)

    horizontal_pattern_decimal_2D = np.tile(horizontal_pattern_decimal[:, np.newaxis], (1, 1280))
    horizontal_pattern_decimal_2D_RGB = convert_7bit_to_rgb(horizontal_pattern_decimal_2D)

    vertical_pattern_decimal_2D = np.tile(vertical_pattern_decimal, (720, 1))
    vertical_pattern_decimal_2D_RGB = convert_7bit_to_rgb(vertical_pattern_decimal_2D)

    cv2.imshow("horizontal_pattern_decimal_2D_RGB", horizontal_pattern_decimal_2D_RGB)
    cv2.imshow("vertical_pattern_decimal_2D_RGB", vertical_pattern_decimal_2D_RGB)
    cv2.imwrite("horizontal_pattern_decimal_2D_RGB.png", horizontal_pattern_decimal_2D_RGB)
    cv2.imwrite("vertical_pattern_decimal_2D_RGB.png", vertical_pattern_decimal_2D_RGB)
    cv2.waitKey(0)


    horizontal_images_path = glob.glob("/home/piljae/Dataset/Hubitz/calibration/patternSetChoose/1/horizontal/*")
    vertical_images_path = glob.glob("/home/piljae/Dataset/Hubitz/calibration/patternSetChoose/1/vertical/*")

    h_result_binary, v_result_binary = structured_light_info_of_image(horizontal_images_path, vertical_images_path)

    h_result_RGB_images = convert_7bit_to_rgb(h_result_binary)
    v_result_RGB_images = convert_7bit_to_rgb(v_result_binary)

    cv2.imshow("h_result_RGB_images", h_result_RGB_images)
    cv2.imshow("v_result_RGB_images", v_result_RGB_images)
    cv2.imwrite("h_result_RGB_images.png", h_result_RGB_images)
    cv2.imwrite("v_result_RGB_images.png", v_result_RGB_images)
    cv2.waitKey(0)