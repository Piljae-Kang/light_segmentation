import cv2
import numpy as np
import glob


angle = 0
expose = 500

folder_path = glob.glob(f"/media/piljae/X31/Dataset/Hubitz/exposure_test/cs{angle}/{expose}_{expose}_{expose}/new_patterns/*")

breakpoint()

import os

os.makedirs(f"light_test/cs{expose}/{angle}", exist_ok=True)

folder_path.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

images = []
for image_path in folder_path:

    image = cv2.imread(image_path)
    image = image/255.0
    images.append(image)


image1 = images[21]
image2 = images[22]


scale = 30
cv2.imshow(f"1", image1)
cv2.imshow(f"2", image2)
cv2.imshow(f"Magic", np.abs(image1 - image2) * scale)
cv2.imwrite(f"light_test/cs{expose}/{angle}/frequency1.png", np.clip(np.abs(image1 - image2) * scale * 255, 0, 255))
cv2.waitKey(0)


cv2.imwrite(f"light_test/cs{expose}/{angle}/frequency2.png", np.clip(np.abs(images[23] - images[24]) * scale * 255, 0, 255))
cv2.imwrite(f"light_test/cs{expose}/{angle}/frequency3.png", np.clip(np.abs(images[25] - images[26]) * scale * 255, 0, 255))
cv2.imwrite(f"light_test/cs{expose}/{angle}/frequency4.png", np.clip(np.abs(images[8] - images[10]) * scale * 255, 0, 255))