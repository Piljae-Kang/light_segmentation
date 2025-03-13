import numpy as np
import cv2
import glob

def apply_gamma(image, max_value, gamma=1.0):

    corrected = np.power(image/max_value, gamma)
    return corrected


material = "metal_bell"
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

cv2.imshow("0 shifted", images[0])
cv2.waitKey(0)
cv2.imshow("45 shifted", images[1])
cv2.waitKey(0)
cv2.imshow("225 shifted", images[5])
cv2.waitKey(0)

max_image_0_1 = np.max(images[0:2, :, :], axis=0)
min_image_0_6 = np.minimum(images[0], images[6])
min_image_0_2 = np.minimum(images[0], images[2])

cv2.imshow("max 0/45 shifted", max_image_0_1)
cv2.imshow("min_image_0_6", min_image_0_6)
cv2.imshow("min_image_0_2", min_image_0_2)
cv2.waitKey(0)

cv2.imshow("L_plus", np.maximum(min_image_0_6, min_image_0_2))
cv2.imshow("L_minus", np.minimum(min_image_0_6, min_image_0_2))

L_d = np.maximum(min_image_0_6, min_image_0_2) - np.minimum(min_image_0_6, min_image_0_2)
L_g = np.minimum(min_image_0_6, min_image_0_2)
L_d_max = np.max(L_d)
L_g_max = np.max(L_g)

if L_d_max > L_g_max:
    max_value = L_d_max
else:
    max_value = L_g_max

cv2.imshow("L_plus", np.maximum(min_image_0_6, min_image_0_2))
cv2.imshow("L_minus", np.minimum(min_image_0_6, min_image_0_2))


cv2.imshow('L_d', apply_gamma(L_d, max_value, 1/2.2))
cv2.imshow('L_g', apply_gamma(L_g, max_value, 1/2.2))
cv2.imshow('L_d + L_g', L_d + L_g)
cv2.imshow('image0', images[0])
cv2.waitKey(0)

shifted_45_index = images[1] < images[5]

modified_image1 = images[1].copy()
modified_image1[shifted_45_index] = 0

shifted_225_index = images[1] > images[5]

modified_image5 = images[5].copy()
modified_image5[shifted_225_index] = 0

cv2.imshow("modified_image1", modified_image1)
cv2.waitKey(0)

cv2.imshow("modified_image5", modified_image5)
cv2.waitKey(0)


cv2.imshow("manipulated_quarter_pattern", max_image_0_1 - modified_image1 + modified_image5)
cv2.waitKey(0)

breakpoint()