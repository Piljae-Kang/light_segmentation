import glob
import numpy as np
import cv2
import os

material = "metal_shaft"
path = f"/media/piljae/X31/Dataset/Hubitz/depth_compute/original_images/{material}/3"
images_path = glob.glob(f"{path}/*.png")
images_path.sort()

output_foler = f"{path}_cropped"
os.makedirs(output_foler, exist_ok=True)

for i, image_path in enumerate(images_path):

    img = cv2.imread(image_path)

    img[180:280, 255:340] = 0

    cv2.imwrite(f"{output_foler}/{i}.png", img)