import cv2
import glob
import re
import argparse
import numpy as np


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    args = parser.parse_args()

    mode = args.mode

    ################ original images

    # material = "metal_shaft"
    # images_path = glob.glob(f"/media/piljae/X31/Dataset/Hubitz/depth_compute/original_images/{material}/1/*")
    # output_path = f"/media/piljae/X31/Dataset/Hubitz/depth_compute/direct_16_pattern/original_images"

    # import os
    # os.makedirs(output_path, exist_ok=True)

    ##############################################

    ########################  exposure test images

    # angle = 45
    # expose = 1000

    # folder_path = f"/media/piljae/X31/Dataset/Hubitz/exposure_test/cs{angle}/{expose}_{expose}_{expose}"
    # images_path = glob.glob(f"{folder_path}/depth_images/*")

    # import os

    # new_folder = f"{folder_path}/replay_depth_images_{mode}_{args.start}_{args.end}"
    # os.makedirs(new_folder, exist_ok=True)
    # output_path = new_folder

    ###########################

    ######################## direct 16 patterns median

    # filtering = "median"
    # folder_name = "direct_16_pattern_scale2"
    # kernel_size = 5
    # alpha = 29
    # images_path = glob.glob(f"/media/piljae/X31/light_segmentation/src/{folder_name}/{filtering}_filtering_{kernel_size}x{kernel_size}/alpha_{alpha}_beta_1/*")
    # output_path = f"/media/piljae/X31/Dataset/Hubitz/depth_compute/{folder_name}/{filtering}_filtering_{kernel_size}x{kernel_size}/alpha_{alpha}"

    # import os
    
    # os.makedirs(output_path, exist_ok=True)

    ######################################################

    '''
    ####################### 
    import os
    #folder_name = "adaptive_multi_anlges/adaptive_alpha_direct_16_pattern_1_cs45"
    folder_name = "pattern_images/adaptive_alpha_gold_crown_1"
    kernel_size = 3

    alpha_list = []
    alpha_list.append(5)
    alpha_list.append(10)
    alpha_list.append(20)
    #alpha_list.append(29)
    alpha_list.append(30)
    alpha_list.append(50)

    filtering_list = ["gaussian", "median", "median_gaussian", "none"]

    for filtering in filtering_list:

        for alpha in alpha_list:

            if filtering == "median":
                images_path = glob.glob(f"/media/piljae/X31/VScode/light_segmentation/src/{folder_name}/{filtering}_filtering_{kernel_size}x{kernel_size}/alpha_{alpha}_beta_1/*")
                output_path = f"/media/piljae/X31/Dataset/Hubitz/depth_compute/{folder_name}/{filtering}_filtering_{kernel_size}x{kernel_size}/alpha_{alpha}"
            else:
                images_path = glob.glob(f"/media/piljae/X31/VScode/light_segmentation/src/{folder_name}/{filtering}_filtering/alpha_{alpha}_beta_1/*")
                output_path = f"/media/piljae/X31/Dataset/Hubitz/depth_compute/{folder_name}/{filtering}_filtering/alpha_{alpha}"

            os.makedirs(output_path, exist_ok=True)

            #######################################################

            images_path.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

            images = []

            # breakpoint()

            for image_path in images_path:

                img = cv2.imread(image_path)
                images.append(img)


            if mode == 1: # duplicate all image

                cnt = 0
                for iteration in range(200):

                    for image in images:

                        cv2.imwrite(f"{output_path}/{cnt}.png", image)
                        cnt += 1

                print(cnt)

            if mode == 2: # copy image into specific frame others are black image

                start = args.start
                end = args.end

                cnt = 0

                h, w = np.array(images[0]).shape[:2]
                black_image = np.zeros((h, w))

                for iteration in range(200):
                    
                    
                    if iteration > start and iteration <= end:


                        for image in images:

                            cv2.imwrite(f"{output_path}/{cnt}.png", image)
                            cnt += 1
                    
                    else:

                        for _ in range(len(images)):

                            cv2.imwrite(f"{output_path}/{cnt}.png", black_image)
                            cnt += 1

    '''

    # duplicate 1frame image
    
        
    frame_idx = 15

    import os
    images_path = glob.glob(f"/media/piljae/X31/Dataset/Hubitz/depth_compute/frame_data/gold_crown/half_syn_pattern/4/*")
    output_path = f"/media/piljae/X31/Dataset/Hubitz/depth_compute/syn_test/gold_crown/4/half_syn_pattern/frame_{frame_idx}"

    os.makedirs(output_path, exist_ok=True)

    #######################################################

    images_path.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

    images = []

    # breakpoint()


    for image_path in images_path[frame_idx * 21 : (frame_idx +1) * 21]:

        img = cv2.imread(image_path)
        images.append(img)


    if mode == 1: # duplicate all image

        cnt = 0
        for iteration in range(200):

            for image in images:

                cv2.imwrite(f"{output_path}/{cnt}.png", image)
                cnt += 1

        print(cnt)

    if mode == 2: # copy image into specific frame others are black image

        start = args.start
        end = args.end

        cnt = 0

        h, w = np.array(images[0]).shape[:2]
        black_image = np.zeros((h, w))

        for iteration in range(200):
            
            
            if iteration > start and iteration <= end:


                for image in images:

                    cv2.imwrite(f"{output_path}/{cnt}.png", image)
                    cnt += 1
            
            else:

                for _ in range(len(images)):

                    cv2.imwrite(f"{output_path}/{cnt}.png", black_image)
                    cnt += 1