import cv2
import glob
import numpy as np

img_path = glob.glob("/media/piljae/X31/Dataset/Hubitz/pattern_info/*")
img_path.sort()

imgs = []
for path in img_path:

    img = cv2.imread(path)
    img = (img / 255.0)[:, :, 0]
    imgs.append(img)

new_pattern = []

new_img1 = np.zeros_like(imgs[0])

for i in range(720//9):

    if i % 4 == 0:
        new_img1[4 + 9 * i : 4 + 9 * (i+1)] = 1


new_img2 = np.roll(new_img1, shift=9, axis=0)
new_img3 = np.roll(new_img2, shift=9, axis=0)
new_img4 = np.roll(new_img3, shift=9, axis=0)

new_pattern = [new_img1, new_img2, new_img3, new_img4]

for i in range(4):
    img = imgs[i]

    new_img = new_pattern[i] + new_pattern[(i + 1) % 4]

    if np.array_equal(img, new_img):
        print(f"{i}")
        # 255를 곱하여 정수화
        image_255 = (new_pattern[i] * 255).astype(np.uint8)

        # (720, 1) -> (720, 1, 3) 변환 (RGB 채널 복제)
        rgb_image = np.repeat(image_255, 3, axis=1).reshape(720, 1, 3)

        # cv2.imwrite(f"{i}_new_pattern.bmp", rgb_image)


    cv2.imshow("new_pattern[i]", new_pattern[i])
    cv2.imshow("new_pattern[(i + 1) % 4]", new_pattern[(i + 1) % 4])
    cv2.imshow("new_img", new_img)
    cv2.waitKey(0)