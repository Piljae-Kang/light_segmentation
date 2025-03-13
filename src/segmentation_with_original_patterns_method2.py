import cv2
import numpy as np
import glob

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

type = 0
img_type = shift_45_images

for i in range(len(img_type)):

    up_index = (i - 1) % 4
    down_index = (i + 1) % 4


    up = np.minimum(shift_0_images[0], shift_0_images[3])
    down = np.minimum(shift_0_images[0], shift_45_images[1])

    upup = np.minimum(up, shift_45_images[3])
    updown = np.minimum(up, shift_45_images[0])

    downup = np.minimum(down, shift_45_images[0])
    downdown = np.minimum(down, shift_45_images[3])

    cv2.imshow("up", up)
    cv2.waitKey(0)
    cv2.imshow("down", down)
    cv2.waitKey(0)
    cv2.imshow("upup", upup)
    cv2.waitKey(0)
    cv2.imshow("updown", updown)
    cv2.waitKey(0)

    images = np.stack([upup, updown, downup, downdown])
    max_image = np.max(images, axis=0)
    min_image = np.min(images, axis=0)

    d = max_image - min_image

    cv2.imshow("max_image", max_image)
    cv2.imshow("min_image", min_image)
    cv2.imshow("d", 2 *d)
    cv2.imshow("g", min_image)
    cv2.waitKey(0)