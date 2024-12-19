import numpy as np
import cv2
import glob
from checkerboard_detector import checkerboard_conv_method

# index 1 이미지만을 가지고 일단 checkerboard 만들거임
horizontal_images_path = glob.glob("/home/piljae/Dataset/Hubitz/calibration/patternSetChoose/1/horizontal/*")
vertical_images_path = glob.glob("/home/piljae/Dataset/Hubitz/calibration/patternSetChoose/1/vertical/*")


horizontal_images_path = sorted(horizontal_images_path, key=lambda x: int(x.split('/')[-1].split('-')[-1][:3]))
vertical_images_path = sorted(vertical_images_path, key=lambda x: int(x.split('/')[-1].split('-')[-1][:3]))

import torch.nn.functional as F
import torch

def median_filter_4d(tensor, kernel_size=3,isnp=False, horizontal = False):

    if isnp:
       # np -> tensor
       tensor = torch.asarray(tensor)
       tensor = tensor.permute([0,3,1,2])

    tensor = tensor.float()

    padding = kernel_size // 2
    b, c, h, w = tensor.shape

    # Unfold the tensor to get sliding windows
    unfolded = F.unfold(tensor.view(b * c, 1, h, w), kernel_size, padding=padding)
    unfolded = unfolded.reshape(b, c, kernel_size , kernel_size, -1)
    if  horizontal:
        unfolded = unfolded [:,:,padding,:,:]
    else: 
        unfolded = unfolded [:,:,:,padding,:]
    # Apply median filtering
    median = unfolded.median(dim=2)[0]

    # Fold back to original shape
    median = median.view(b, c, h, w)

    #median = median.byte()

    if isnp:
       #tensor -> no
       median = median.permute([0,2,3,1])
       median = median.numpy()
    return median


def avg_filter_4d(tensor, kernel_size=3,isnp=False, horizontal = False):

    if isnp:
       # np -> tensor
       tensor = torch.asarray(tensor)
       tensor = tensor.permute([0,3,1,2])

    tensor = tensor.float()

    padding = kernel_size // 2
    b, c, h, w = tensor.shape

    # Gaussian Kernel 
    # kernelsize
    # g [].reshape (1,1,kernelsize,-1)
    # g / np.sum(g)
    # (g*unfolded).mean()


    # Unfold the tensor to get sliding windows
    unfolded = F.unfold(tensor.view(b * c, 1, h, w), kernel_size, padding=padding)
    unfolded = unfolded.reshape(b, c, kernel_size , kernel_size, -1)
    if  horizontal:
        unfolded = unfolded [:,:,padding,:,:]
    else: 
        unfolded = unfolded [:,:,:,padding,:]
    # Apply median filtering
    median = unfolded.mean(dim=2)[0]

    # Fold back to original shape
    median = median.view(b, c, h, w)

    #median = median.byte()

    if isnp:
       #tensor -> no
       median = median.permute([0,2,3,1])
       median = median.numpy()
    return median


def max_filter_4d(tensor, kernel_size=3,isnp=False, horizontal = False):

    if isnp:
       # np -> tensor
       tensor = torch.asarray(tensor)
       tensor = tensor.permute([0,3,1,2])

    tensor = tensor.float()

    padding = kernel_size // 2
    b, c, h, w = tensor.shape

    # Unfold the tensor to get sliding windows
    unfolded = F.unfold(tensor.view(b * c, 1, h, w), kernel_size, padding=padding)
    unfolded = unfolded.reshape(b, c, kernel_size , kernel_size, -1)
    if  horizontal:
        unfolded = unfolded [:,:,padding,:,:]
    else: 
        unfolded = unfolded [:,:,:,padding,:]
    # Apply median filtering
    median = unfolded.max(dim=2)[0]

    # Fold back to original shape
    median = median.view(b, c, h, w)

    #median = median.byte()

    if isnp:
       #tensor -> no
       median = median.permute([0,2,3,1])
       median = median.numpy()
    return median


def min_filter_4d(tensor, kernel_size=3,isnp=False, horizontal = False):

    if isnp:
       # np -> tensor
       tensor = torch.asarray(tensor)
       tensor = tensor.permute([0,3,1,2])

    tensor = tensor.float()

    padding = kernel_size // 2
    b, c, h, w = tensor.shape

    # Unfold the tensor to get sliding windows
    unfolded = F.unfold(tensor.view(b * c, 1, h, w), kernel_size, padding=padding)
    unfolded = unfolded.reshape(b, c, kernel_size , kernel_size, -1)
    if  horizontal:
        unfolded = unfolded [:,:,padding,:,:]
    else: 
        unfolded = unfolded [:,:,:,padding,:]
    # Apply median filtering
    median = unfolded.min(dim=2)[0]

    # Fold back to original shape
    median = median.view(b, c, h, w)

    #median = median.byte()

    if isnp:
       #tensor -> no
       median = median.permute([0,2,3,1])
       median = median.numpy()
    return median
def color_to_gray(image):

    return np.dot(image[...,:3], [0.299, 0.587, 0.114])

gamma = True
h_img0s = []
h_img1s = []
white_imgs = []
black_imgs = []



h_img_white = cv2.imread(horizontal_images_path[-2])
h_img_white = h_img_white.astype(np.float32) / 255.0

if gamma:

    h_img_white = h_img_white **(2.4)


for i in range(0, len(horizontal_images_path), 2):
    
    h_img_0 = cv2.imread(horizontal_images_path[i])
    h_img_1 = cv2.imread(horizontal_images_path[i+1])
    v_img_0 = cv2.imread(vertical_images_path[i])
    v_img_1 = cv2.imread(vertical_images_path[i+1])
    (h,w,c) = h_img_0.shape

    # cv2.imshow("h_img_0", h_img_0)
    # cv2.imshow("h_img_1", h_img_1)
    # cv2.waitKey(0)
    # h, w = np.array(img_0).shape

    h_img_0 = h_img_0.astype(np.float32) / 255.0
    h_img_1 = h_img_1.astype(np.float32) / 255.0
    v_img_0 = v_img_0.astype(np.float32) / 255.0
    v_img_1 = v_img_1.astype(np.float32) / 255.0

    #refined_pattern0, refined_pattern1 = clarify ( pattern0, pattern1)
    #refined_pattern0, refined_pattern1 = clarify ( pattern0, pattern1)

    if gamma:

        h_img_0 = h_img_0 **(2.4)
        h_img_1 = h_img_1 **(2.4)

    h_abs_diff_image = np.abs(h_img_0 - h_img_1)
    h_black = np.min( np.concatenate([h_img_0[...,None], h_img_1[...,None]], axis = -1), axis =-1)
    h_white = np.max( np.concatenate([h_img_0[...,None], h_img_1[...,None]], axis = -1), axis =-1)

    h_img0s.append ( h_img_0)
    h_img1s.append ( h_img_1)
    white_imgs.append( h_white)
    black_imgs.append( h_black)

#########################################################
    #img = (h_img_0 -h_black ) / h_abs_diff_image

print()
logScale=False
for base_idx in range (7):
    for idx in range(base_idx):

        bpts = []

        for temp in range(2):

            if temp == 0:
                h_img = h_img0s[idx]
            else:
                h_img = h_img1s[idx]

            if logScale:
                pt_img = (h_img/black_imgs[base_idx]) 
                pt_img = np.log (pt_img)
                pt_img = pt_img - np.min(pt_img)
                pt_img = pt_img / np.max(pt_img)
            else:
                pt_img = (h_img-black_imgs[base_idx]) / (white_imgs[base_idx]-black_imgs[base_idx])
            
            #pt_img [pt_img <0.5]= 0
            #pt_img [pt_img >0.5]= 1
            if gamma:
                pt_img = pt_img ** (1./2.4)
            bpts.append(pt_img)
            filename="pattern"

            if logScale:
                filename += "_logscale"
            if gamma:
                filename += "_gamma"            

            filename = filename+"_base{1:d}_id{0:d}_{2:d}".format(idx, base_idx, temp)
                
            cv2.imwrite( filename+".png", pt_img*255)


        absdiff =np.abs(bpts [1] - bpts[0])
        absdiff= avg_filter_4d(absdiff[None], 3, isnp=True)[0]
        #absdiff = absdiff / np.max(absdiff)

        filename="bdiff"

        if logScale:
            filename += "_logscale"
        if gamma:
            filename += "_gamma"            

        filename = filename+"_base{1:d}_id{0:d}".format(idx, base_idx)
            
        cv2.imwrite( filename+".png", absdiff*255)

#cv2.waitKey(0)


