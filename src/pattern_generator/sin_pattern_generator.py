import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


import matplotlib
matplotlib.use("TkAgg")

def create_sinwave(period, N=720):

    t = np.arange(N)
    wave_01 = 0.5 * (1.0 + np.sin(2.0 * np.pi * (t / period)))

    return wave_01

def shift_sinwave(wave, shift):

    new_wave = np.roll(wave, shift=shift, axis=0)

    return new_wave

img_path = glob.glob("/media/piljae/X31/Dataset/Hubitz/pattern_info/*")
img_path.sort()

imgs = []
for path in img_path:

    img = cv2.imread(path)
    img = (img / 255.0)[:, :, 0].flatten()
    imgs.append(img)

wave = create_sinwave(36)

wave_0 = shift_sinwave(wave, shift=4)
wave_1 = shift_sinwave(wave_0, shift=9)
wave_2 = shift_sinwave(wave_0, shift=18)
wave_3 = shift_sinwave(wave_0, shift=27)

wave_imgs = [wave_0, wave_1, wave_2, wave_3]

for i in range(4):

    wave = wave_imgs[i].reshape(-1, 1)
    img = imgs[i]

    plt.figure(figsize=(10, 4))
    plt.plot(wave, label="Sine wave (0~1), period=18")
    plt.plot(img, label="Sine wave (0~1), period=18")
    plt.ylim([-0.2, 1.2])
    plt.legend()
    plt.title("720x1 samples with 18-sample period sine wave")
    plt.show()



    wave_255 = (wave * 255).astype(np.uint8)

    # (720, 1) -> (720, 1, 3) 변환 (RGB 채널 복제)
    rgb_wave = np.repeat(wave_255, 3, axis=1).reshape(720, 1, 3)
    rgb_wave_1080 = np.repeat(rgb_wave, 1080, axis=1)

    cv2.imwrite(f"{i}_high_sin_pattern.bmp", rgb_wave_1080)