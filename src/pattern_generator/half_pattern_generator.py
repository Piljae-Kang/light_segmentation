import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


import matplotlib
matplotlib.use("TkAgg")

def shift_sinwave(wave, shift):

    new_wave = np.roll(wave, shift=shift, axis=0)

    return new_wave

img_path = glob.glob("/media/piljae/X31/Dataset/Hubitz/pattern_info/original_hubitz/*")
img_path.sort()

imgs = []
for path in img_path:

    img = cv2.imread(path)
    img = (img / 255.0)[:, :, 0].flatten()
    imgs.append(img)

half_pattern_0 = np.zeros_like(imgs[0])

gap_size = 9

for i in range(720//gap_size):

    if i % 4 == 0:
        half_pattern_0[gap_size * i : gap_size * (i+1)] = 1

half_pattern_45 = shift_sinwave(half_pattern_0, shift=4)

cv2.imshow("half_pattern_0", half_pattern_0)
cv2.imshow("half_pattern_45", half_pattern_45)
cv2.waitKey(0)

half_pattern_0_list = []
half_pattern_45_list = []

for i in range(4):

    shift_val = i * 9

    half_pattern_0_shift = shift_sinwave(half_pattern_0, shift=shift_val)
    half_pattern_45_shift = shift_sinwave(half_pattern_45, shift=shift_val)

    half_pattern_0_list.append(half_pattern_0_shift)
    half_pattern_45_list.append(half_pattern_45_shift)

    cv2.imshow("half_pattern_0_shift", half_pattern_0_shift)
    cv2.imshow("half_pattern_45_shift", half_pattern_45_shift)
    cv2.waitKey(0)

    half_pattern_0_shift_255 = (half_pattern_0_shift * 255).astype(np.uint8)
    half_pattern_45_shift_255 = (half_pattern_45_shift * 255).astype(np.uint8)


    # (720, 1) -> (720, 1, 3) 변환 (RGB 채널 복제)
    rgb_half_pattern_0_shift_255 = np.repeat(half_pattern_0_shift_255.reshape(-1, 1), 3, axis=1).reshape(720, 1, 3)
    rgb_half_pattern_0_shift_255_1080 = np.repeat(rgb_half_pattern_0_shift_255, 1080, axis=1)
    
    rgb_half_pattern_45_shift_255 = np.repeat(half_pattern_45_shift_255.reshape(-1, 1), 3, axis=1).reshape(720, 1, 3)
    rgb_half_pattern_45_shift_255_1080 = np.repeat(rgb_half_pattern_45_shift_255, 1080, axis=1)

    cv2.imwrite(f"{i}_shift0.png", rgb_half_pattern_0_shift_255_1080)
    cv2.imwrite(f"{i}_shift45.png", rgb_half_pattern_45_shift_255_1080)


## 맞는지 확인하는 과정

# imgs_0 = imgs[12:16]
# imgs_45 = imgs[0:4]


# for i in range(4):

#     idx1 = i
#     idx2 = (i+1)%4

#     sum_0 = half_pattern_0_list[i] + half_pattern_0_list[(i+1)%4]
#     sum_45 = half_pattern_45_list[i] + half_pattern_45_list[(i+1)%4]

#     print(np.array_equal(sum_0, imgs_0[i]))
#     print(np.array_equal(sum_45, imgs_45[i]))