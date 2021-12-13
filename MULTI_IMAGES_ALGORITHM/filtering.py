"""
Group Members:
    Akkapelli Harshith
    Tammireddy Sasank 
    (Ordered by First Letter)
"""

#import necessary libraries
import numpy as np
import cv2
import numpy as np
from scipy import ndimage as ndi
import skimage.segmentation


class filtering:
    """
    Purpose: it does median filtering and makes things smoother.
    parameters:
        image: input image
        param: parameter for median filtering
    """
    def __init__(self,image,param):
        self.img   = image
        self.param = param
    
    def routine(self):
        new_img = cv2.medianBlur(self.img,self.param)
        return new_img

"""
Testing 
"""
# im = cv2.imread('out3_3.png')
# fil = filtering(im,5)
# out = fil.routine()
# cv2.imwrite('out4.png',out)