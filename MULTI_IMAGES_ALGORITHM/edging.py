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

class edging:
    """
    Purpose: to detect edges in image using canny edge algorithm
    parameters:
        img: input image
        param1,param2: parameters of canny edge
    """
    def __init__(self,img,param1,param2):
        self.img    = img
        self.param1 = param1
        self.param2 = param2
    
    def routine(self):
        edg = cv2.Canny(self.img,self.param1,self.param2)
        return edg 

class hough:
    """
    Purpose: to detect hough lines in image using hough transform
    parameters:
        img: input image
        param1,param2: parameters of hough transform
    """
    def __init__(self,img,param1,param2):
        self.img    = img
        self.param1 = param1
        self.param2 = param2 
    
    def sine(self,param):
        return np.sin(param)
    
    def cose(self,param):
        return np.cos(param)
    
    def get_x_y(self,param3,param4):
        return (int(param3 - 1000*param4),int(param3 + 1000*param4))
    
    def get_x0_y0(self,param3,param4,param5):
        return (int(param3*param5),int(param4*param5))
    
    def routine(self):
        lines = cv2.HoughLines(self.img,self.param1,np.pi/180,self.param2,np.array([]), 0, 0)
        new_img = np.zeros(self.img.shape)
        for i in range(len(lines)):
            for r,theta in lines[i]:
                a = self.cose(theta)        
                b = self.sine(theta)
                x0,y0 = self.get_x0_y0(a,b,r)                                    
                x1,x2 = self.get_x_y(x0,b)
                y2,y1 = self.get_x_y(y0,a)
                cv2.line(new_img,(x1,y1), (x2,y2), (255,255,255),1)
        return new_img

"""
Testing 
"""
# im = cv2.imread('out4.png')
# img = np.zeros(im.shape)
# v = np.median(im)
# sigma = 0.33
# lower_thresh = int(max(0, (1.0 - sigma) * v))
# upper_thresh = int(min(255, (1.0 + sigma) * v))
# edge = edging(im,lower_thresh,upper_thresh)
# out = edge.routine()
# cv2.imwrite('out5_1.png',out)

# hou = hough(out,1,120)
# out = hou.routine()
# print(out.shape)
# cv2.imwrite('out5_2.png',out)