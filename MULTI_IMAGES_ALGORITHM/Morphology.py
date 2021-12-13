"""
Group Members:
    Akkapelli Harshith
    Tammireddy Sasank 
    (Ordered by First Letter)
"""

#import libraries
import numpy as np
import cv2
import numpy as np



class ImageErosion:
    """
    Purpose: performs Image erosion
    parameters:
        image: input image
        kernel: input kernel
        max_iter: number of iteartions
        threshold: input threshold
    """
    def __init__(self, image, kernel, max_iter=1, threshold=128, binarize=True):
        self.image = image.copy()
        self.kernel = kernel
        self.iters = max_iter
        self.threshold = threshold
        self.binarize = binarize
        
    ## utility functions for ease
    def getImage(self):
        return self.image
    
    def getKernel(self):
        return self.kernel
    
    def getThreshold(self):
        return self.threshold
    
    def setKernel(self, kernel):
        self.kernel = kernel
        return self.kernel
    
    def setThreshold(self, eta):
        self.threshold = eta
        return self.threshold
    
    def getKernelSize(self):
        return self.kernel.shape
    
    def getImageSize(self):
        return self.image.shape
    
    def transform(self):
        if(self.binarize):
            self.image[self.image>self.threshold] = 255
            self.image[self.image<=self.threshold] = 0
        out = cv2.erode(self.image, self.kernel, iterations=self.iters)
        out[out>0] = 255
        return out
    
class ImageDilation:
    """
    Purpose: performs Image dilation
    parameters:
        image: input image
        kernel: input kernel
        max_iter: number of iteartions
        threshold: input threshold
    """
    def __init__(self, image, kernel, max_iter=1, threshold=128, binarize=True):
        self.image = image.copy()
        self.kernel = kernel
        self.iters = max_iter
        self.threshold = threshold
        self.binarize = binarize
        
    ## utility functions for ease
    def getImage(self):
        return self.image
    
    def getKernel(self):
        return self.kernel
    
    def getThreshold(self):
        return self.threshold
    
    def setKernel(self, kernel):
        self.kernel = kernel
        return self.kernel
    
    def setThreshold(self, eta):
        self.threshold = eta
        return self.threshold
    
    def getKernelSize(self):
        return self.kernel.shape
    
    def getImageSize(self):
        return self.image.shape
    
    def transform(self):
        if(self.binarize):
            self.image[self.image>self.threshold] = 255
            self.image[self.image<=self.threshold] = 0
        out = cv2.dilate(self.image, self.kernel, iterations=self.iters)
        out[out>0] = 255
        return out
    
class Inversion:
    """
    Purpose: performs inversion of image which is nothing but making white pixels black and vice versa
    parameters:
        image: input image
    """
    def __init__(self,  image):
        self.image = image.copy()
        
    def transform(self):
        out = self.image.copy()
        out[self.image>0], out[self.image<=0] = 0, 255
        return out
    
"""
TESTING
"""

# img = cv2.imread('out2.jpg', 0)
# k2 = np.zeros((101,101), np.uint8)
# k2[50, 50] = 1
# k1 = np.ones((3,3), np.uint8) 

# t1 = ImageDilation(img, k1)
# t1 = t1.transform()
# cv2.imwrite('out3_1.png', t1)

# t2 = Inversion(t1)
# t2 = t2.transform()
# cv2.imwrite('out3_2.png', t2)    

# t3 = ImageErosion(t2, k2)
# t3 = t3.transform()
# cv2.imwrite('out3_3.png', t3)    