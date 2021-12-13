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

class corner:
    """
    purpose: Detecting corners in a image
    Parameters:
        img: input image
        param1,param2,param3,param4: parameters of our corner detection class
    """
    def __init__(self,img,param1,param2,param3,param4):
        self.img    = img
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3
        self.param4 = param4
    
    def routine(self):
        new_img = cv2.cornerHarris(self.img,self.param1,self.param2,self.param3)
        new_img = self.use_func(new_img)
        #new_img = cv2.goodFeaturesToTrack(new_img, 150, 0.6, 1)
       # new_img = np.int0(new_img)
        #print(new_img)
        #new_img = self.use_(new_img)
        return new_img
    
    def use_func(self,im):
        new_img = np.zeros(self.img.shape)
        ll = []
        kkk = set()
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                if(im[i][j] > 0.35*im.max()):
                    ll.append((i,j))
        for i in range(len(ll)):
            if(ll[i] not in kkk):
                for j in range(i+1,len(ll)):
                    if(ll[j] not in kkk):
                        if(((ll[i][0]-ll[j][0])*(ll[i][0]-ll[j][0])+(ll[i][1]-ll[j][1])*(ll[i][1]-ll[j][1])) <= self.param4):

                            kkk.add(ll[j])
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                if((i,j) not in kkk and im[i][j] > 0.35*im.max()):
                    new_img[i][j] = 255       
        return new_img
    
    def use_(self,out):
        new_im = np.zeros(self.img.shape)
        for i in range(len(out)):
            for j in range(len(out[0])):
                new_im[int(out[i][j][0]),int(out[i][j][1])] = 255
        return new_im

"""
Testing
"""
# im = cv2.imread("out5_2.png",0)
# #im= cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
# #im = np.array(im,dtype="float32")
# cor = corner(im,2,3,0.01,144)
# out = cor.routine()

# cv2.imwrite("out6.png",out)