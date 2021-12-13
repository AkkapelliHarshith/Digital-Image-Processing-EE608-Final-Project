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



class segmentation:
    """
    Purpose: This class  implements segmentation techniques such as:
        1) Mean shift segmentation - slow as thriple nested for loop is involved and no built in is available in opencv python
        2) k means based segmentation
    parameters:
        image: input image
        param: input parameter
        num_iterations: number of iterations for mean shift segmentation
        spatial radius: spatial radius for mean shift segmentation
        range_radius: range radius for mean shift segmentation
        min_density: minimum density for mean shift segmentation
    """
    def __init__(self, image, param, num_iterations,spatial_radius,range_radius,min_density):
        self.image          = image
        self.param          = param
        self.num_iterations = num_iterations
        self.image_height   = image.shape[0]
        self.image_width    = image.shape[1]
        self.spatial_radius = spatial_radius
        self.range_radius   = range_radius
        self.min_density    = min_density
                            
    def image_copying(self):
        return self.image.copy()
    
    def image_square(self,img):
        return np.square(img)
    
    def image_sum_param(self,img,p):
        return img.sum(p)
    
    def image_sum(self,img):
        return img.sum()
    
    def image_shift(self,img,shif):
        return (img - shif)
    
    def image_div_cons(self,img,l):
        return img/l
    
    def image_exp(self,l):
        return np.exp(-l)

    def routine_mean_shift(self):
        temp  = self.image_copying()
        temp_ = self.image_copying()
        
        for iter_ in range(self.num_iterations):
            for pixel_y in range(self.image_height):
                for pixel_x in range(self.image_width):
                    temp_1 = self.image_shift(temp,temp[pixel_y,pixel_x])
                    temp_1 = self.image_square(temp_1)
                    temp_1 = self.image_sum_param(temp_1,-1)
                    temp_1 = self.image_exp(temp_1)
                    
                    temp_2 = ((temp-temp[pixel_y, pixel_x]) * np.expand_dims(temp_1, -1)).sum((0, 1))
                    
                    temp_3 = temp_2/temp_1.sum()
                    temp_[pixel_y,pixel_x] += temp_3
            temp = temp_
            print('Progress: {:.03f}{}\n'.format(100*(iter_*self.image_height+pixel_y+1)/self.num_iterations/self.image_height, '%'), end='\r')
        return temp
    
    def routine_kmeans(self):
        temp = self.routine2()
        new_img = self.routine3(temp)
        return new_img
    
    def routine2(self):
        vv = self.image.reshape((-1,3))
        vv = np.float32(vv)        
        return vv
    
    def routine3(self,im):
        ite = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 2
        t,b,r=cv2.kmeans(im,K,None,ite,10,cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(r)
        ss = r[b.flatten()]
        ss2 = ss.reshape((self.image.shape))       
        return ss2
    """
    def optimized_mean_shift(self):
        my_segmenter = pms.Segmenter()

        my_segmenter.spatial_radius = self.spatial_radius  #6
        my_segmenter.range_radius = self.range_radius #4.5
        my_segmenter.min_density = self.min_density #50        
        (segmented_image, labels_image, number_regions) = my_segmenter(self.image)
        r
    """



"""
Testing of mean shift segmentation
"""
# im = cv2.imread('temp.jpg')
# seg = segmentation(im,0.1,100,6,4.5,50)
# out = seg.optimized_mean_shift()
# cv2.imwrite('out1.png',out)