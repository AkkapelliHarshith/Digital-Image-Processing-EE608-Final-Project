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
from skimage.segmentation import *
import skimage.segmentation

class masking:
    """
    Purpose:This class is useful for making mask on the image such that:
        there are only two types of pixel black or white
    Parameters:
        img: input image
        start_point: starting point for flood fill
        start_value: value to fill
    """
    def __init__(self,img,start_point,start_value):
        self.img         = img
        self.start_point = start_point
        self.start_value = start_value
    
    def routine(self):
        temp1 = skimage.segmentation.flood_fill(self.img[:,:,0], self.start_point,self.start_value)
        temp2 = skimage.segmentation.flood_fill(self.img[:,:,1], self.start_point,self.start_value)
        temp3 = skimage.segmentation.flood_fill(self.img[:,:,2], self.start_point,self.start_value)
        temp = np.zeros(self.img.shape)
        temp[:,:,0] = temp1
        temp[:,:,1] = temp2
        temp[:,:,2] = temp3
        
        temp = self.use_func(temp)
        return temp
        
    def use_func(self,temp):
        for i in range(temp.shape[0]):
            for j in range(temp.shape[1]):
                for k in range(temp.shape[2]):
                    if(temp[i,j,k] != 255):
                        temp[i,j,k] = 0
                        
        return temp

"""
Testing
"""
# from segmentation import *
# im = cv2.imread('test.jpg')
# seg = segmentation(im,0.1,100,6,4.5,50)
# out = seg.routine_kmeans()
# mask = masking(out,(0,0),255)
# out  = mask.routine()
# cv2.imwrite('out2.jpg',out) 