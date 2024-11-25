import cv2
import numpy as np

# size = (720, 1280)
size = (720, 1280) # projector
# vertical_pattern = np.zeros(size)
# horizontal_pattern = np.zeros(size)

vertical_edge = np.zeros(size)
horizontal_edge = np.zeros(size)
edge = np.zeros(size)

#bwidth = 5
#phase_gaps= [ bwidth * (2**x) for x in range (9) ]
phase_gaps = [8, 12, 20]
case_sizes = [2, 4]



for phase_gap in phase_gaps:
        
    for case in range(6):

        horizontal_pattern = np.zeros(size)

        gap = int(phase_gap / 2)

        for i in range((size[0]//phase_gap) + 1):

            if i % 2 == 0:


                if case<3:
                    horizontal_pattern[phase_gap * i + case * int(gap/2)  : phase_gap * i + case * int(gap/2) + gap, :] = 255
                if case == 3:
                    horizontal_pattern[phase_gap * i + case * int(gap/2)  : phase_gap * (i+1), :] = 255
                    horizontal_pattern[phase_gap * i  : phase_gap * i + int(gap/2), :] = 255

                if case == 4:
                    horizontal_pattern[phase_gap * i + 0 * int(gap/2)  : phase_gap * i + 1 * int(gap/2), :] = 255
                    horizontal_pattern[phase_gap * i + 2 * int(gap/2)  : phase_gap * i + 3 * int(gap/2), :] = 255

                if case == 5:
                    horizontal_pattern[phase_gap * i + 1 * int(gap/2)  : phase_gap * i + 2 * int(gap/2) :] = 255
                    horizontal_pattern[phase_gap * i + 3 * int(gap/2)  : phase_gap * (i+1), :] = 255


        #breakpoint()
        #cv2.imshow(f"new_pattern/horizontal_pattern_gap{phase_gap}_{case}_original.png", horizontal_pattern)
        #cv2.waitKey(0)
        cv2.imwrite(f"new_pattern/horizontal_pattern_gap{phase_gap}_{case}_original.png", horizontal_pattern[:, 0:1])

    #cv2.imwrite(f"horizontal_pattern_gap{phase_gap}_{case}.png", horizontal_pattern[:, 0:1])
    #cv2.imwrite(f"new_pattern/horizontal_pattern_gap{phase_gap}_{case}_original.png", horizontal_pattern[:, 0:1])

breakpoint()

for case_size in case_sizes:


    for phase_gap in phase_gaps:

        horizontal_pattern = np.zeros(size)

        for i in range((size[0]//phase_gap) + 1):

                #horizontal_edge[phase_gap * i, :] = 255


            if i % 2 == 0:

                horizontal_pattern[phase_gap * i  : phase_gap * (i+1), :] = 255

        
        #cv2.imshow(f"pattern/case_size_{case_size}/horizontal_pattern_gap{phase_gap}_{case_size}_original.png", horizontal_pattern)
        #cv2.waitKey(0)

        # cv2.imwrite(f"horizontal_pattern_gap{phase_gap}_{case}.png", horizontal_pattern[:, 0:1])
        cv2.imwrite(f"pattern/case_size_{case_size}/horizontal_pattern_gap{phase_gap}_{case_size}_original.png", horizontal_pattern[:, 0:1])

        for case in range(0, case_size):

            horizontal_pattern = np.zeros(size)

            

            for i in range((size[0]//phase_gap) + 1):

                #horizontal_edge[phase_gap * i, :] = 255


                if i % 2 == 0:

                    horizontal_pattern[phase_gap * i + int(phase_gap/case_size) * case  : phase_gap * i + int(phase_gap/case_size) * (case+1), :] = 255




            #cv2.imshow(f"pattern/case_size_{case_size}/horizontal_pattern_gap{phase_gap}_{case_size}_{case}.png", horizontal_pattern)
            #cv2.waitKey(0)

            # cv2.imwrite(f"horizontal_pattern_gap{phase_gap}_{case}.png", horizontal_pattern[:, 0:1])
            cv2.imwrite(f"pattern/case_size_{case_size}/horizontal_pattern_gap{phase_gap}_{case_size}_{case}.png", horizontal_pattern[:, 0:1])



        # horizontal_pattern = np.zeros(size)

        # for i in range((size[0]//phase_gap) + 1):

        #         #horizontal_edge[phase_gap * i, :] = 255


        #     if i % 2 == 0:

        #         horizontal_pattern[phase_gap * i  : phase_gap * (i+1), :] = 255

        
        # cv2.imshow(f"pattern/case_size_{case_size}/horizontal_pattern_gap{phase_gap}_{case_size}_original.png", horizontal_pattern)
        # cv2.waitKey(0)

        # # cv2.imwrite(f"horizontal_pattern_gap{phase_gap}_{case}.png", horizontal_pattern[:, 0:1])
        # cv2.imwrite(f"pattern/case_size_{case_size}/horizontal_pattern_gap{phase_gap}_{case_size}_original.png", horizontal_pattern[:, 0:1])
