import numpy as np
import cv2
import glob

def apply_gamma(image, max_value, gamma=1.0):

    corrected = np.power(image/max_value, gamma)
    return corrected


def sin2square(A, B, C, D):

    condition = ((A - C) > 0) & ((B - D) > 0)

    g = np.maximum(A, B) + np.minimum(C, D) - np.full_like(A, np.max(np.maximum(A, C)))

    d = np.maximum(A, B) + np.minimum(C, D) - g

    image = g

    image[condition] = d[condition]

    # cv2.imshow("g image", g)
    # cv2.waitKey()
    # cv2.imshow("d image", d)
    # cv2.waitKey()
    # cv2.imshow("image", image)
    # cv2.waitKey()

    return image

def clear_image(images, type, multiply=1):
    A = images[0] # 0 shift 
    B = images[1] # 90 shift
    C = images[2] # 180 shift
    D = images[3] # 270 shift

    image_1 = sin2square(A,B,C,D)

    A = images[3] # 0 shift 
    B = images[0] # 90 shift
    C = images[1] # 180 shift
    D = images[2] # 270 shift

    image_2 = sin2square(A,B,C,D)

    A = images[2] # 0 shift 
    B = images[3] # 90 shift
    C = images[0] # 180 shift
    D = images[1] # 270 shift

    image_3 = sin2square(A,B,C,D)

    A = images[1] # 0 shift 
    B = images[2] # 90 shift
    C = images[3] # 180 shift
    D = images[0] # 270 shift

    image_4 = sin2square(A,B,C,D)


    # phase 0
    L_plus = np.maximum(image_1, image_2)
    L_minus = np.minimum(image_1, image_2)

    # cv2.imshow("L_plus", L_plus)
    # cv2.imshow("L_minus", L_minus)
    #cv2.imshow("d", multiply*( L_plus - L_minus))
    cv2.imwrite(f"{type * 4}.png", np.clip((multiply*( L_plus - L_minus))* 255, 0, 255))
    cv2.waitKey(0)

    #phase 1
    L_plus = np.maximum(image_2, image_3)
    L_minus = np.minimum(image_2, image_3)

    # cv2.imshow("L_plus", L_plus)
    # cv2.imshow("L_minus", L_minus)
    #cv2.imshow("d", multiply*( L_plus - L_minus))
    cv2.imwrite(f"{type * 4 + 1}.png", np.clip((multiply*( L_plus - L_minus))* 255, 0, 255))
    cv2.waitKey(0)

    #phase 2
    L_plus = np.maximum(image_3, image_4)
    L_minus = np.minimum(image_3, image_4)

    # cv2.imshow("L_plus", L_plus)
    # cv2.imshow("L_minus", L_minus)
    #cv2.imshow("d", multiply*( L_plus - L_minus))
    cv2.imwrite(f"{type * 4 + 2}.png", np.clip((multiply*( L_plus - L_minus)) * 255, 0, 255))
    cv2.waitKey(0)

    #phase 3
    L_plus = np.maximum(image_4, image_1)
    L_minus = np.minimum(image_4, image_1)

    # cv2.imshow("L_plus", L_plus)
    # cv2.imshow("L_minus", L_minus)
    #cv2.imshow("d", multiply*( L_plus - L_minus))
    cv2.imwrite(f"{type * 4 + 3}.png", np.clip((multiply*( L_plus - L_minus))* 255, 0, 255))
    cv2.waitKey(0)


material = "metal_bell"
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


clear_image(shift_0_images, 0, multiply=3)
clear_image(low_0_images, 1, multiply=3)
clear_image(middle_0_images, 2, multiply=3)
clear_image(shift_45_images, 3, multiply=3)