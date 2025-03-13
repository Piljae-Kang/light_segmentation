import os

def linux2window(path):
    
    path_window = path.split("/")
    
    del path_window[:4]
    
    path_window.insert(0, "D:")
    
    return "\\".join(path_window)


path = "/media/piljae/X31/Dataset/Hubitz/depth_compute/method1_segmentation_images/correct"

path = linux2window(path)
print(path)