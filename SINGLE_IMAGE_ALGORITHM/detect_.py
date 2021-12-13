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



class detect_:
    """
    Purpose: to detect protraits in given image
    parameters:
        img: input image
        orig_img: original image
    """
    def __init__(self,img,orig_img):
        self.img = img
        self.orig_img = orig_img
    
    def routine(self):
        min_x,max_x,min_y,max_y = self.get_coord()
        new_im = np.zeros((-min_x+max_x+1,-min_y+max_y+1,3))
        new_im[:,:,:] = self.orig_img[min_x:max_x+1,min_y:max_y+1,:]
        return new_im 
    
    def get_coord(self):
        coordinates = []
        for i in range(len(self.img)):
            for j in range(len(self.img[0])):
                if(self.img[i,j] == 255.0):
                    coordinates.append(np.array([i,j]))
        min_x = 9999
        max_x = -1
        min_y = 99999
        max_y = 1
        
        for i in range(len(coordinates)):
            min_x = min(min_x,coordinates[i][0])
            min_y = min(min_y,coordinates[i][1])
            max_x = max(max_x,coordinates[i][0])
            max_y = max(max_y,coordinates[i][1])

        return (min_x,max_x,min_y,max_y)
    
"""
Testing
"""

# im = cv2.imread('out6.png',0)
# orig_img = cv2.imread('temp.jpg')
# det_pain = detect_protraits(im,orig_img)
# out = det_pain.routine()
# cv2.imwrite('out7.png',out)