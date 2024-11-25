import numpy as np
import cv2
import glob

def apply_gamma(image, max_value, gamma=1.0):

    corrected = np.power(image/max_value, gamma)
    return corrected


material = "metal_shaft"
phase_gap = 8
images_path = glob.glob(f"/home/piljae/Dataset/Hubitz/light_path_segmentation/8patterns/{material}/gap_{phase_gap}/*")
images_path.sort()

images = []
for image_path in images_path:

    print(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    images.append(img)

images = np.array(images)

A = images[0] # 0 shift 
B = images[2] # 90 shift
C = images[4] # 180 shift
D = images[6] # 270 shift

condition = ((A - C) > 0) & ((B - D) > 0)

g = np.maximum(A, B) + np.minimum(C, D) - np.full_like(A, np.max(np.maximum(A, C)))

d = np.maximum(A, B) + np.minimum(C, D) - g

image = g

image[condition] = d[condition]

cv2.imshow("g image", g)
cv2.waitKey()
cv2.imshow("d image", d)
cv2.waitKey()
cv2.imshow("image", image)
cv2.waitKey()




breakpoint()