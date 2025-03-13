import cv2
import glob
import numpy as np
import argparse

# command : python segmentation_with_pattern.py --root_path /home/piljae/Dataset/Hubitz/light_path_segmentation/scan_images --material metal_bell --output_root_path /home/piljae/Dropbox/hubitz/light_segmentation/segmentation_result

def apply_gamma(image, max_value, gamma=1.0):

    corrected = np.power(image/max_value, gamma)
    return corrected


def quarter_4pattern(path, output_path, original_patterns):

    images_path = glob.glob(f"{path}/*")
    images_path.sort()

    level = int(len(images_path)/4)

    pattern_4_images = []
    images = []

    for i, image_path in enumerate(images_path):

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
        images.append(img)

        if (i+1) % 4 == 0:
            pattern_4_images.append(images)
            images = []
    
    pattern_4_images = np.array(pattern_4_images)

    for i in range(pattern_4_images.shape[0]):

        current_frequency_images = pattern_4_images[i, :, :, :]

        original_patern_image = original_patterns[i]

        sorted_data = np.sort(current_frequency_images, axis=0)[::-1]

        L_plus = np.max(current_frequency_images, axis=0)
        L_minus = np.min(current_frequency_images, axis=0)
        #L_minus = (sorted_data[2,:,:] + sorted_data[3,:,:]) / 2
        #L_minus = sorted_data[2, :, :]

        L_plus_max = np.max(L_plus)
        L_minus_max = np.max(L_minus)

        if L_plus_max > L_minus_max:
            max_value = L_plus_max
        else:
            max_value = L_minus_max

        cv2.imshow('L_plus', (apply_gamma(L_plus, L_plus_max, 1/2.2)))
        cv2.imwrite(f'{output_path}/L_plus.png', (L_plus * 255).astype(np.uint8)) 

        cv2.imshow('L_minus', (apply_gamma(L_minus, L_minus_max, 1/2.2)))
        cv2.imwrite(f'{output_path}/L_minus.png', (L_minus * 255).astype(np.uint8))

        L_d = L_plus - L_minus
        L_g = 4 * L_minus

        L_d_max = np.max(L_d)
        L_g_max = np.max(L_g)
        
        if L_d_max > L_g_max:
            max_value = L_d_max
        else:
            max_value = L_g_max

        cv2.imshow('L_d', apply_gamma(L_d, max_value, 1/2.2))
        cv2.imwrite(f'{output_path}/L_d.png', (apply_gamma(L_d, max_value, 1/2.2) * 255).astype(np.uint8))
        cv2.imwrite(f'{output_path}/L_d_multiply4.png', (4 * L_d * 255).astype(np.uint8))

        cv2.imshow('L_g', apply_gamma(L_g, max_value, 1/2.2))
        cv2.imwrite(f'{output_path}/L_g.png', (apply_gamma(L_g, max_value, 1/2.2) * 255).astype(np.uint8))

        L_ = L_d + L_g
        pixel_diff = np.abs(L_ - original_patern_image)

        # max_total = np.max(L_)
        # max_original = np.max(original_patern_image)

        # if max_total > max_original:
        #     max_value = max_total
        # else:
        #     max_value = max_original

        cv2.imshow('original_pattern_image', apply_gamma(original_patern_image, max_value, 1/2.2))
        cv2.imwrite(f'{output_path}/original_pattern_image.png', (apply_gamma(original_patern_image, max_value, 1/2.2) * 255).astype(np.uint8))


        cv2.imshow(f'L_d + L_g : {np.mean(pixel_diff)}', apply_gamma(L_, max_value, 1/2.2))
        cv2.imwrite(f'{output_path}/L_d_plus_L_g.png', (L_ * 255).astype(np.uint8))
        cv2.waitKey(0)


        break # 첫번째 frequency만 저장함

    

def half_4pattern(path, output_path, original_patterns):

    images_path = glob.glob(f"{path}/*")
    images_path.sort()

    level = int(len(images_path)/4)

    half_4pattern_images = []
    images = []

    for i, image_path in enumerate(images_path):

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
        images.append(img)

        if (i+1) % 4 == 0:
            half_4pattern_images.append(images)
            images = []
    
    half_4pattern_images = np.array(half_4pattern_images)

    for i in range(half_4pattern_images.shape[0]):

        current_frequency_images = half_4pattern_images[i, :, :, :]

        original_patern_image = original_patterns[i]

        L_plus = np.max(current_frequency_images, axis=0)
        L_minus = np.min(current_frequency_images, axis=0)

        cv2.imshow('L_plus', L_plus)
        cv2.imwrite(f'{output_path}/L_plus.png', (L_plus * 255).astype(np.uint8)) 

        cv2.imshow('L_minus', L_minus)
        cv2.imwrite(f'{output_path}/L_minus.png', (L_minus * 255).astype(np.uint8))

        L_d = L_plus - L_minus
        L_g = 2 * L_minus

        L_d_max = np.max(L_d)
        L_g_max = np.max(L_g)
        
        if L_d_max > L_g_max:
            max_value = L_d_max
        else:
            max_value = L_g_max


        cv2.imshow('L_d', apply_gamma(L_d, max_value, 1/2))
        cv2.imwrite(f'{output_path}/L_d.png', (apply_gamma(L_d, max_value, 1/2.2) * 255).astype(np.uint8))
        cv2.imwrite(f'{output_path}/L_d_multiply4.png', (4 * L_d * 255).astype(np.uint8))

        cv2.imshow('L_g', apply_gamma(L_g, max_value, 1/2))
        cv2.imwrite(f'{output_path}/L_g.png', (apply_gamma(L_g, max_value, 1/2.2) * 255).astype(np.uint8))

        cv2.imshow('original_pattern_image', original_patern_image)
        cv2.imwrite(f'{output_path}/original_pattern_image.png', (apply_gamma(original_patern_image,max_value, 1/2.2) * 255).astype(np.uint8))

        L_ = L_d + L_g
        pixel_diff = np.abs(L_ - original_patern_image)
        cv2.imshow(f'L_d + L_g : {np.mean(pixel_diff)}', L_d + L_g)
        cv2.imwrite(f'{output_path}/L_d_plus_L_g_pixel_diff_{pixel_diff}.png', (L_ * 255).astype(np.uint8))
        cv2.waitKey(0)


        break # 첫번째 frequency만 저장함

def half_2pattern(path, output_path, original_patterns):

    folders = glob.glob(f"{path}/*")
    folders.sort()

    for folder_path in folders:

        folder_name = folder_path.split("/")[-1]

        images_path = glob.glob(f"{folder_path}/*")
        images_path.sort()

        half_2pattern_images = []
        images = []

        for i, image_path in enumerate(images_path):

            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32) / 255.0
            images.append(img)

            if (i+1) % 2 == 0:
                half_2pattern_images.append(images)
                images = []
        
        half_2pattern_images = np.array(half_2pattern_images)

        for i in range(half_2pattern_images.shape[0]):

            current_frequency_images = half_2pattern_images[i, :, :, :]

            original_patern_image = original_patterns[i]

            L_plus = np.max(current_frequency_images, axis=0)
            L_minus = np.min(current_frequency_images, axis=0)

            cv2.imshow('L_plus', L_plus)
            cv2.imwrite(f'{output_path}/{folder_name}/L_plus.png', (L_plus * 255).astype(np.uint8)) 

            cv2.imshow('L_minus', L_minus)
            cv2.imwrite(f'{output_path}/{folder_name}/L_minus.png', (L_minus * 255).astype(np.uint8))

            L_d = L_plus - L_minus
            L_g = 2 * L_minus


            L_d_max = np.max(L_d)
            L_g_max = np.max(L_g)
            
            if L_d_max > L_g_max:
                max_value = L_d_max
            else:
                max_value = L_g_max

            cv2.imshow('L_d', 4 *L_d)
            cv2.imwrite(f'{output_path}/{folder_name}/L_d.png', (apply_gamma(L_d, max_value, 1/2.2) * 255).astype(np.uint8))
            cv2.imwrite(f'{output_path}/{folder_name}/L_d_multiply4.png', (4 * L_d * 255).astype(np.uint8))

            cv2.imshow('L_g', L_g)
            cv2.imwrite(f'{output_path}/{folder_name}/L_g.png', (apply_gamma(L_g, max_value, 1/2.2) * 255).astype(np.uint8))

            cv2.imshow('original_pattern_image', original_patern_image)
            cv2.imwrite(f'{output_path}/{folder_name}/original_pattern_image.png', (apply_gamma(original_patern_image, max_value, 1/2.2) * 255).astype(np.uint8))

            L_ = L_d + L_g
            pixel_diff = np.abs(L_ - original_patern_image)
            cv2.imshow(f'L_d + L_g : {np.mean(pixel_diff)}', L_d + L_g)
            cv2.imwrite(f'{output_path}/{folder_name}/L_d_plus_L_g_pixel_diff_{pixel_diff}.png', (L_ * 255).astype(np.uint8))
            cv2.waitKey(0)

            break # 첫번째 frequency만 저장함
        






def compute_rgb_image(path, output_path):

    R_image_path = glob.glob(f"{path}/R/*")[0]
    G_image_path = glob.glob(f"{path}/G/*")[0]
    B_image_path = glob.glob(f"{path}/B/*")[0]

    R_img = cv2.imread(R_image_path, cv2.IMREAD_GRAYSCALE)
    G_img = cv2.imread(G_image_path, cv2.IMREAD_GRAYSCALE)
    B_img = cv2.imread(B_image_path, cv2.IMREAD_GRAYSCALE)

    RGB_img = cv2.merge([B_img, G_img, R_img])

    cv2.imshow("RGB Image", RGB_img)
    cv2.imwrite(f"{output_path}/rgb.png", RGB_img)
    cv2.waitKey(0)



def original_pattern(path, output_path):

    images_path = glob.glob(f"{path}/*")
    images_path.sort()

    images = []
    for i, image_path in enumerate(images_path):
        
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
        images.append(img)
    
    return images


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path", type=str)
    parser.add_argument("--material", type=str)
    parser.add_argument("--output_root_path", type=str)

    args = parser.parse_args()

    image_path = f"{args.root_path}/{args.material}"
    output_path = f"{args.output_root_path}/{args.material}"

    original_patterns = original_pattern(image_path + "/original_pattern", output_path + "/original_pattern")

    quarter_4pattern(image_path + "/quarter_4pattern", output_path + "/quarter_4pattern", original_patterns)
    half_4pattern(image_path + "/half_4pattern", output_path + "/half_4pattern", original_patterns)
    half_2pattern(image_path + "/half_2pattern", output_path + "/half_2pattern", original_patterns)
    compute_rgb_image(image_path + "/color_img", output_path + "/color_img")


