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

import time

time1 = time.time()
im = cv2.imread('test.jpg')

"""
Mean shift segmentation
"""
from segmentation import segmentation
seg = segmentation(im,0.1,100,6,4.5,50)
out = seg.routine_kmeans()
cv2.imwrite('results/out1.png',out)

"""
Masking
"""
from masking import *
mask = masking(out,(0,0),255)
out  = mask.routine()
cv2.imwrite('results/out2.jpg',out) 

"""
Morphology
"""
from Morphology import ImageDilation,ImageErosion,Inversion
img = cv2.imread('results/out2.jpg', 0)
k2 = np.zeros((101,101), np.uint8)
k2[50, 50] = 1
k1 = np.ones((3,3), np.uint8) 

t1 = ImageDilation(img, k1)
t1 = t1.transform()
cv2.imwrite('results/out3_1.png', t1)

t2 = Inversion(t1)
t2 = t2.transform()
cv2.imwrite('results/out3_2.png', t2)    

t3 = ImageErosion(t2, k2)
t3 = t3.transform()
cv2.imwrite('results/out3_3.png', t3)

"""
Filtering
"""
from filtering import  filtering
im = cv2.imread('results/out3_3.png')
fil = filtering(im,3)
out = fil.routine()
cv2.imwrite('results/out4.png',out)

"""
Edge detection and Hough lines detection
"""
from edging import edging,hough
im = cv2.imread('results/out4.png')
img = np.zeros(im.shape)
v = np.median(im)
sigma = 0.33
lower_thresh = int(max(0, (1.0 - sigma) * v))
upper_thresh = int(min(255, (1.0 + sigma) * v))
edge = edging(im,lower_thresh,upper_thresh)
out = edge.routine()
cv2.imwrite('results/out5_1.png',out)

hou = hough(out,1,60)
out = hou.routine()
cv2.imwrite('results/out5_2.png',out)

"""
Corner detection
"""
from corners import corner
im = cv2.imread("results/out5_2.png",0)
#im= cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
#im = np.array(im,dtype="float32")
cor = corner(im,2,3,0.01,144)
out = cor.routine()

cv2.imwrite("results/out6.png",out)

"""
Detect protraits
"""
from detect_ import detect_
im = cv2.imread('results/out6.png',0)
orig_img = cv2.imread('test.jpg')
det_pain = detect_(im,orig_img)
out = det_pain.routine()
cv2.imwrite('results/detected/detected.png',out)


"""
Final image matching and evaluation of our model
"""
from final_img_matching_and_evaluation import final_img_matching_and_evaluation
img1 = cv2.imread('results/detected/detected.png',cv2.IMREAD_GRAYSCALE)          
img2 = cv2.imread('results/ground/ground.png',cv2.IMREAD_GRAYSCALE) 


fin = final_img_matching_and_evaluation(img1,img2)
scor = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED).max()
out = fin.routine_1()
cv2.imwrite('results/out7.png',out)

time2 = time.time()


print("################## Results ##################")
print("Time taken by our algorithm is: "+str(time2-time1)+" seconds")
print("Total number of protraits: ",1)
print("Our TM_CCOEFF_NORMED coefficient is %.2f"%(np.array(scor).mean()))
print("#################Conclusion##################")
print("for results and images please see results folder in the directory in which this file is present")
print("So our algorithm is promising and good enough")