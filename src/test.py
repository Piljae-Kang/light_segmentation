import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter, gaussian_filter
import matplotlib.pyplot as plt

# Filtering function
def filtering(image, parameter=3, type=0, axis=None):
    if type == 0:  # Median filtering
        return median_filter(image, size=parameter)
    
    elif type == 1:  # Gaussian filtering
        if axis == "vertical":
            # Apply Gaussian filter only in the vertical direction (axis=0)
            return gaussian_filter1d(image, sigma=parameter, axis=0)
        elif axis == "horizontal":
            # Apply Gaussian filter only in the horizontal direction (axis=1)
            return gaussian_filter1d(image, sigma=parameter, axis=1)
        else:
            # Apply 2D Gaussian filter as usual
            return gaussian_filter(image, sigma=parameter)
    
    else:
        return image
    
    
def create_horizontal_pattern_image(width=256, height=256, pattern_frequency=10):
    """
    Creates a grayscale image with horizontal patterns.
    """
    image = np.zeros((height, width), dtype=np.uint8)
    for i in range(0, height, pattern_frequency):
        image[i:i + pattern_frequency // 2] = 255  # Create horizontal white lines
    return image


# Create a horizontal pattern image
pattern_image = create_horizontal_pattern_image()

# Apply vertical Gaussian filtering
vertical_filtered_image = filtering(pattern_image, parameter=1, type=1, axis="vertical")

# Apply vertical Gaussian filtering
filtered_image = filtering(pattern_image, parameter=1, type=1)

cv2.imshow("pattern_image", pattern_image)
cv2.imshow("vertical_filtered_image", vertical_filtered_image)
cv2.imshow("filtered_image", filtered_image)
cv2.waitKey(0)