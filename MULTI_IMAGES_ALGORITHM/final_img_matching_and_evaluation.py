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
from skimage.metrics import structural_similarity


class final_img_matching_and_evaluation:
    """
    Purpose: it matches input images with corresponding output images and evaluates it based on:
         cv2.TM_CCOEFF_NORMED
    parameters:
        imgs: actual images
        detc: detected images
    """
    def __init__(self,imgs,detc):
        self.imgs = imgs
        self.detc = detc
    
    def routine_1(self):
        map_ = dict()
        accuracy = []
        for det in self.detc:
            im = cv2.imread(det)
            scor = -1
            fil = ""
            for name in self.imgs:
                img = cv2.imread(name)
                if(cv2.matchTemplate(img, im, cv2.TM_CCOEFF_NORMED ).max() > scor):
                    scor = cv2.matchTemplate(img, im, cv2.TM_CCOEFF_NORMED).max()
                    fil = name
            map_[det] = fil
            accuracy.append(scor)
        
        return (map_,accuracy)
    
class final_img_matching_and_evaluation_:
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
            if m.distance < 0.5*n.distance:
                o.append([m])
        return o
    
    def routine_5(self,p1,p2,ee):
        new_img = cv2.drawMatchesKnn(self.img1,p1,self.img2,p2,ee,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return new_img

"""
TESTING
"""        

# imgs = ['ground_truth_0.png','ground_truth_1.png','ground_truth_2.png','ground_truth_3.png','ground_truth_4.png','ground_truth_5.png','ground_truth_6.png','ground_truth_7.png','ground_truth_8.png']
# detc = ['detected_0.png','detected_1.png','detected_2.png','detected_3.png','detected_4.png','detected_5.png','detected_6.png','detected_7.png','detected_8.png']
# fin1 = final_img_matching_and_evaluation(detc,imgs)
# map_,acc = fin1.routine_1()
# print("Our Average accuracy is %.2f%s"%(np.array(acc).mean()*100,'%'))
# j = 0
# for key in map_:
#     a = cv2.imread(key)
#     b = cv2.imread(map_[key])
#     fin2 = final_img_matching_and_evaluation_(a,b)
#     out = fin2.routine_1()
#     fil = "match_"+str(j)+".jpg"
#     cv2.imwrite(fil,out)
#     j+=1