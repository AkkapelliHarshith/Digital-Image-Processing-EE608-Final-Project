
#import necessary libraries
import numpy as np
import cv2
import numpy as np
from scipy import ndimage as ndi
import skimage.segmentation



class finding: 
    """
    Purpose: finds sub imagess in a given image
    parameters:
        img: input image
        orig_img: original image
    """
    def __init__(self,img,orig_img):
        self.img = img
        self.orig_img = orig_img
    
    def routine_1(self):
        contours, hierarchy = cv2.findContours(self.img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(cnt) for cnt in contours]
        rects = set(rects)
        return rects 
    
    def routine_2(self,rects):
        temp = np.zeros(self.img.shape)
        for i in rects:
            x,y,w,h = i
            temp[ y:h+y,x:w+x] = 255 
        return temp
    
    def routine_3(self,rects):
        j = 0
        for i in rects:
            x,y,w,h = i
            temp = self.orig_img[ y:h+y,x:w+x,:]
            fil = 'results/detected/detected_'+str(j)+'.png'
            cv2.imwrite(fil,temp)
            j+=1
        return len(rects)

"""
TESTING
"""
# img = cv2.imread('out5_1.png',0)
# orig_img =cv2.imread('temp.jpg')
# fin_pain = finding(img,orig_img)
# rects = fin_pain.routine_1()
# det = fin_pain.routine_2(rects)
# te = np.zeros(img.shape)
# for i in rects:
    
#     x,y,w,h = i
#     te[ y:h+y,x:w+x] = 255 
# cv2.imwrite('out7.png',te)
# n = fin_pain.routine_3(rects)
# print(n)