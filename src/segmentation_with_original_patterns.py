import cv2
import glob
import numpy as np

def apply_gamma(image, max_value, gamma=1.0):

    corrected = np.power(image/max_value, gamma)
    return corrected

material = "metal_shaft"
images_path = glob.glob(f"/media/piljae/X31/Dataset/Hubitz/depth_compute/original_images/{material}/1/*")
images_path.sort()

images = []
for image_path in images_path:
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    images.append(img)

images = np.array(images)

shift_0_images = images[:4]
low_0_images = images[4:8]
middle_0_images = images[8:12]
shift_45_images = images[12:16]

type = 3
img_type = shift_45_images

# for i in range(len(img_type)):
#     cv2.imshow("image_", img_type [i] )
#     cv2.waitKey(0)


for i in range(len(img_type)):

    up_index = (i - 1) % 4
    down_index = (i + 1) % 4

    # breakpoint()

    image_up = np.minimum(img_type[i], img_type[up_index])
    image_down = np.minimum(img_type[i], img_type[down_index])
    image =img_type[i]

    cv2.imshow("min_image_0_3", image_up)
    cv2.imshow("min_image_0_1", image_down)
    cv2.waitKey(0)

    L_plus = np.maximum(image_up, image_down)
    L_minus = np.minimum(image_up, image_down)

    cv2.imshow("L_plus", L_plus)
    cv2.imshow("L_minus", L_minus)

    L_d = L_plus - L_minus
    L_g = L_minus

    L_d_max = np.max(L_d)
    L_g_max = np.max(L_g)

    if L_d_max > L_g_max:
        max_value = L_d_max
    else:
        max_value = L_g_max

    cv2.imshow('L_d', apply_gamma(L_d, max_value, 1/2.2))
    cv2.imshow('L_g', apply_gamma(L_g, max_value, 1/2.2))
    cv2.imshow("multiply", 2 * L_d)
    cv2.imshow('L_d + L_g', L_d + L_g)
    cv2.waitKey(0)

    cv2.imwrite(f"{type * 4 + i}.png", L_d* 255)