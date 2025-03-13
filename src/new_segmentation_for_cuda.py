import cv2
import numpy as np
import glob
from scipy.ndimage import median_filter, gaussian_filter, gaussian_filter1d

def filtering(image, parameter=3, type=0):
    
    if type == 0:
        return median_filter(image, size=parameter)
    
    if type == 1:
        return gaussian_filter(image, sigma=parameter)
    
    if type == 2:
        return gaussian_filter1d(image, sigma=parameter, axis=1)
    
    else:
        return image

# Sigmoid 함수
def sigmoid(x, alpha=1.0):
    return 1 / (1 + np.exp(-alpha * x))


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
rgb_images = images[16:]

import os

filtering_type = "none"

filterings = ["median", "gaussian", "none", "vertical_gaussian"]

kernel_size = 3
gaussian_parameter = 3.0

alpha = 30
beta = 1

for i in range(5):

    if i == 4:
        
        type = i
        rgb_images = images[4 * i:4 * i + 5]

        cv2.imwrite(f"direct_16_pattern/{4*type + 0}.png", (rgb_images[0] * 255).astype(np.uint8))
        cv2.imwrite(f"direct_16_pattern/{4*type + 1}.png", (rgb_images[1] * 255).astype(np.uint8))
        cv2.imwrite(f"direct_16_pattern/{4*type + 2}.png", (rgb_images[2] * 255).astype(np.uint8))
        cv2.imwrite(f"direct_16_pattern/{4*type + 3}.png", (rgb_images[3] * 255).astype(np.uint8))
        cv2.imwrite(f"direct_16_pattern/{4*type + 4}.png", (rgb_images[4] * 255).astype(np.uint8))

        for filtering_type in filterings:
                    
            if filtering_type == "median":
                folder_path = f"direct_16_pattern/{filtering_type}_filtering_{kernel_size}x{kernel_size}/alpha_{alpha}_beta_{beta}"

            else:
                folder_path = f"direct_16_pattern/{filtering_type}_filtering/alpha_{alpha}_beta_{beta}"
            
            os.makedirs(folder_path, exist_ok=True)

            cv2.imwrite(f"{folder_path}/{4*type + 0}.png", (rgb_images[0] * 255).astype(np.uint8))
            cv2.imwrite(f"{folder_path}/{4*type + 1}.png", (rgb_images[1] * 255).astype(np.uint8))
            cv2.imwrite(f"{folder_path}/{4*type + 2}.png", (rgb_images[2] * 255).astype(np.uint8))
            cv2.imwrite(f"{folder_path}/{4*type + 3}.png", (rgb_images[3] * 255).astype(np.uint8))
            cv2.imwrite(f"{folder_path}/{4*type + 4}.png", (rgb_images[4] * 255).astype(np.uint8))


    else:
        type = i
        img_type = images[4*i : 4*i + 4]


        alpha_list = [i for i in range(1, 30, 1)]
        beta_list = [1]

        image1 = img_type[0]
        image2 = img_type[1]
        image3 = img_type[2]
        image4 = img_type[3]

        # cv2.imshow("image1:", image1)
        # cv2.imshow("image2:", image2)
        # cv2.imshow("image3:", image3)
        # cv2.imshow("image4:", image4)
        #cv2.waitKey(0)

        cv2.imwrite(f"direct_16_pattern/{4*type + 0}.png", (image1 * 255).astype(np.uint8))
        cv2.imwrite(f"direct_16_pattern/{4*type + 1}.png", (image2 * 255).astype(np.uint8))
        cv2.imwrite(f"direct_16_pattern/{4*type + 2}.png", (image3 * 255).astype(np.uint8))
        cv2.imwrite(f"direct_16_pattern/{4*type + 3}.png", (image4 * 255).astype(np.uint8))

        

        max_image = np.maximum.reduce([image1, image2, image3, image4])
        min_image = np.minimum.reduce([image1, image2, image3, image4])

        L_d = max_image - min_image

        L_g = 0.5 * min_image



        cv2.imshow("L_d:", L_d)
        cv2.imshow("max_image:", max_image)
        cv2.imshow("min_image:", min_image * 0.5)
        cv2.imshow("image", L_d + L_g)
        cv2.waitKey(0)

        cv2.imwrite("direct_pattern/max_image.png", (max_image * 255).astype(np.uint8))
        cv2.imwrite("direct_pattern/min_image.png", (min_image * 255).astype(np.uint8))
        cv2.imwrite("direct_pattern/L_d.png", (L_d * 255).astype(np.uint8))
        cv2.imwrite("direct_pattern/L_g.png", (L_g * 255).astype(np.uint8))
        cv2.waitKey(0)


        for filtering_type in filterings:

            if filtering_type == "median":
                folder_path = f"direct_16_pattern/{filtering_type}_filtering_{kernel_size}x{kernel_size}/alpha_{alpha}_beta_{beta}"

            else:
                folder_path = f"direct_16_pattern/{filtering_type}_filtering/alpha_{alpha}_beta_{beta}"
            
            os.makedirs(folder_path, exist_ok=True)
            
            w_med = sigmoid(alpha * filtering(image1 - image3, parameter=kernel_size), alpha=beta)
            w_gaus = sigmoid(alpha * filtering(image1 - image3, parameter=gaussian_parameter, type=1), alpha=beta)
            w_none = sigmoid(alpha * filtering(image1 - image3, parameter=gaussian_parameter, type=2), alpha=beta)
            
            if alpha == 1 or alpha == 5 or alpha == 10 or alpha == 20 or alpha == 29:
                cv2.imwrite(f"median_{alpha}.png", w_med*255)
                cv2.imwrite(f"gauss_{alpha}.png", w_gaus*255)
                cv2.imwrite(f"none_{alpha}.png", w_none*255)


            if filtering_type == "median":
                #median filtering
                w_1 = sigmoid(alpha * filtering(image1 - image3, parameter=kernel_size), alpha=beta)
                w_2 = sigmoid(alpha * filtering(image2 - image4, parameter=kernel_size), alpha=beta)
                w_3 = sigmoid(alpha * filtering(image3 - image1, parameter=kernel_size), alpha=beta)
                w_4 = sigmoid(alpha * filtering(image4 - image2, parameter=kernel_size), alpha=beta)

            if filtering_type == "gaussian":
                # gaussian filtering
                w_1 = sigmoid(alpha * filtering(image1 - image3, parameter=gaussian_parameter, type=1), alpha=beta)
                w_2 = sigmoid(alpha * filtering(image2 - image4, parameter=gaussian_parameter, type=1), alpha=beta)
                w_3 = sigmoid(alpha * filtering(image3 - image1, parameter=gaussian_parameter, type=1), alpha=beta)
                w_4 = sigmoid(alpha * filtering(image4 - image2, parameter=gaussian_parameter, type=1), alpha=beta)
                
            if filtering_type == "vertical_gaussian":
                # gaussian filtering
                w_1 = sigmoid(alpha * filtering(image1 - image3, parameter=gaussian_parameter, type=2), alpha=beta)
                w_2 = sigmoid(alpha * filtering(image2 - image4, parameter=gaussian_parameter, type=2), alpha=beta)
                w_3 = sigmoid(alpha * filtering(image3 - image1, parameter=gaussian_parameter, type=2), alpha=beta)
                w_4 = sigmoid(alpha * filtering(image4 - image2, parameter=gaussian_parameter, type=2), alpha=beta)                    

            
            else:
                # none filtering
                w_1 = sigmoid(alpha * filtering(image1 - image3, parameter=1.0, type=2), alpha=beta)
                w_2 = sigmoid(alpha * filtering(image2 - image4, parameter=1.0, type=2), alpha=beta)
                w_3 = sigmoid(alpha * filtering(image3 - image1, parameter=1.0, type=2), alpha=beta)
                w_4 = sigmoid(alpha * filtering(image4 - image2, parameter=1.0, type=2), alpha=beta)

            # cv2.imshow("w_1:", w_1)
            # cv2.imshow("w_2:", w_2)
            # cv2.imshow("w_3:", w_3)
            # cv2.imshow("w_4:", w_4)
            # cv2.waitKey(0)

            I1_ours = w_1 * L_d
            I2_ours = w_2 * L_d
            I3_ours = w_3 * L_d
            I4_ours = w_4 * L_d

            max_value1 = np.max(I1_ours)
            max_value2 = np.max(I2_ours)
            max_value3 = np.max(I3_ours)
            max_value4 = np.max(I4_ours)

            I1_ours = I1_ours/max_value1
            I2_ours = I2_ours/max_value2
            I3_ours = I3_ours/max_value3
            I4_ours = I4_ours/max_value4

            I1_ours = np.clip(I1_ours * 2 , 0, 1)
            I2_ours = np.clip(I2_ours * 2 , 0, 1)
            I3_ours = np.clip(I3_ours * 2 , 0, 1)
            I4_ours = np.clip(I4_ours * 2 , 0, 1)
            
            cv2.imwrite(f"{folder_path}/{4*type + 0}.png", (I1_ours * 255).astype(np.uint8))
            cv2.imwrite(f"{folder_path}/{4*type + 1}.png", (I2_ours * 255).astype(np.uint8))
            cv2.imwrite(f"{folder_path}/{4*type + 2}.png", (I3_ours * 255).astype(np.uint8))
            cv2.imwrite(f"{folder_path}/{4*type + 3}.png", (I4_ours * 255).astype(np.uint8))
            
            print(folder_path)

            # cv2.imshow("I1_ours:", I1_ours)
            # cv2.imshow("I2_ours:", I2_ours)
            # cv2.imshow("I3_ours:", I3_ours)
            # cv2.imshow("I4_ours:", I4_ours)
            # cv2.waitKey(0)