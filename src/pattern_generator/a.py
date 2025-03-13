import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


import matplotlib
matplotlib.use("TkAgg")


gap = 20
path = f"/media/piljae/X31/Dataset/Hubitz/pattern_info/separation_pattern/case_size_4/horizontal_pattern_gap{gap}_4_original.png"

img = cv2.imread(path)
img = (img / 255.0)[:, :, 0].flatten()

for i in range(4):

    name = f"horizontal_pattern_gap{gap}_4_original_{i}.png"

    new_pattern = np.roll(img, shift=int(gap/2 * i), axis=0)

    new_pattern_255 = (new_pattern * 255).astype(np.uint8)


    # (720, 1) -> (720, 1, 3) 변환 (RGB 채널 복제)
    rgb_wave = np.repeat(new_pattern_255.reshape(-1, 1), 3, axis=1).reshape(720, 1, 3)
    rgb_wave_1080 = np.repeat(rgb_wave, 1080, axis=1)
    

    cv2.imwrite(name, rgb_wave_1080)