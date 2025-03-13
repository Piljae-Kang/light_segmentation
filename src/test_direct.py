import cv2
import glob
import numpy as np

def mouse_callback(event, x, y, flags, param):
    # 마우스가 움직일 때마다(MOUSEMOVE) 픽셀 값을 확인
    if event == cv2.EVENT_MOUSEMOVE:
        # B, G, R 순서로 픽셀이 저장되어 있음
        b = img[y, x, 0]
        print(f"X:{x}, Y:{y}, B:{b}")

material = "gold_crown"
case_num = 3
# images_path = glob.glob(f"/media/piljae/X31/Dataset/Hubitz/devided_pattern/{material}/frame/*.png")

images_path = glob.glob(f"/media/piljae/X31/Dataset/Hubitz/depth_compute/original_images/{material}/{case_num}/*.png")
images_path.sort()

images = []
for image_path in images_path:
    
    img = cv2.imread(image_path)
    img = img.astype(np.float32) / 255.0
    images.append(img)


images = np.array(images)

shift_0_images = images[:4]
low_0_images = images[4:8]
middle_0_images = images[8:12]
shift_45_images = images[12:16]
divided_images = images[16:20]

sin_images = images[20:24]

L_d_list = []
L_g_list = []
max_list = []
min_list = []

for i in range(4):

    image1 = shift_0_images[i]
    image2 = shift_0_images[(i+1)%4]

    image3 = divided_images[(i+2)%4]
    image4 = divided_images[(i+3)%4]

    edge = np.zeros_like(image1)

    # cv2.namedWindow("Image")
    # # "Image"라는 창에 콜백 함수 등록
    # cv2.setMouseCallback("Image", mouse_callback)

    # while True:
    #     cv2.imshow("Image", image1)
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == 27:  # ESC 키를 누르면 종료
    #         break



    edge[np.abs(image1 - image3) < 0.1] = 1

    edge = (image1 - image3)

    edge = (edge - np.min(edge)) / np.max(edge - np.min(edge))

    #edge[edge <0.3] = 0

    L_plus = np.maximum.reduce([image1, image2])
    L_minus = np.minimum.reduce([image1, image2])

    L_d = L_plus - L_minus #* np.mean(L_plus)/np.mean(L_plus)
    L_g = 0.5 * L_minus

    # cv2.imshow(f"edge", edge)
    # cv2.imshow(f"L_plus", L_plus)
    # cv2.imshow(f"L_minus", L_minus)
    # cv2.imshow(f"L_d", L_d)
    # cv2.imshow(f"L_g", L_g)
    # cv2.imshow(f"image1 + image2", image1 + image2)
    # cv2.imshow(f"shift_0_images", shift_0_images[i])
    # cv2.imshow(f"image1 + image2 - shift_0_images", image1 + image2 - shift_0_images[i])
    # cv2.waitKey(0)



high_max = np.maximum.reduce([shift_0_images[0], shift_0_images[1], shift_0_images[2], shift_0_images[3]])
low_max = np.maximum.reduce([low_0_images[0], low_0_images[1], low_0_images[2], low_0_images[3]])
high_min = np.minimum.reduce([shift_0_images[0], shift_0_images[1], shift_0_images[2], shift_0_images[3]])
low_min = np.minimum.reduce([low_0_images[0], low_0_images[1], low_0_images[2], low_0_images[3]])


# cv2.imshow("high_max",high_max)
# cv2.imshow("low_max",low_max)
# cv2.imshow("high_min",high_min)
# cv2.imshow("low_min",low_min)
# cv2.imshow("diff", np.abs(high_max - low_max))
# cv2.imshow("high_max - high_min", np.abs(high_max - high_min))
# cv2.imshow("low_max - low_min", np.abs(low_max - low_min))
# cv2.waitKey(0)


# cv2.imshow(high_max - high_min,  