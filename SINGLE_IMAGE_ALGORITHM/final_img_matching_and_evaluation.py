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



class final_img_matching_and_evaluation:
    """
    Purpose: it matches features across two images
    parameters:
        img2: actual images
        img1: detected images
    """
    def __init__(self,img1,img2):
        self.img1 = img1 
        self.img2 = img2

    def routine_2(self,im):
        sift = cv2.SIFT_create()
        a,b = sift.detectAndCompute(im,None)
        return (a,b)
    
    def routine_3(self,im1,im2):
        bf = cv2.BFMatcher()
        temp = bf.knnMatch(im1,im2,k=2)
        return temp
    
    def routine_1(self):        
        p1, s1 = self.routine_2(self.img1)
        p2, s2 = self.routine_2(self.img2)
        rr = self.routine_3(s1,s2)
        ee = self.routine_4(rr)
        new_img = self.routine_5(p1,p2,ee)
        return new_img
       
    def routine_4(self,matches):
        o = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                o.append([m])
        return o
    
    def routine_5(self,p1,p2,ee):
        new_img = cv2.drawMatchesKnn(self.img1,p1,self.img2,p2,ee,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return new_img

"""
Final image matching and evaluation of our model
""" 
# img1 = cv2.imread('out7.png',cv2.IMREAD_GRAYSCALE)          
# img2 = cv2.imread('ground_truth.png',cv2.IMREAD_GRAYSCALE) 


# fin = final_img_matching_and_evaluation(img1,img2)
# scor = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED).max()
# print("Our Average accuracy is %.2f%s"%(scor*100,'%'))
# out = fin.routine_1()
# cv2.imwrite('out8_1.png',out)
