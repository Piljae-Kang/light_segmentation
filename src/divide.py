import cv2
import os
import glob


angle = 45
expose = 1000

folder_path = f"/media/piljae/X31/Dataset/Hubitz/exposure_test/cs{angle}/{expose}_{expose}_{expose}"

paths = glob.glob(f"{folder_path}/*")

paths.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

images = []


for path in paths:

    img = cv2.imread(path)
    images.append(img)


depth_images = images[:21]

pattern_images = images[21:]

os.makedirs(f"{folder_path}/depth_images", exist_ok=True)

for i, image in enumerate(depth_images):

    cv2.imwrite(f"{folder_path}/depth_images/{i}.png", image)


os.makedirs(f"{folder_path}/pattern_images", exist_ok=True)

for i, image in enumerate(pattern_images):

    cv2.imwrite(f"{folder_path}/pattern_images/{i}.png", image)