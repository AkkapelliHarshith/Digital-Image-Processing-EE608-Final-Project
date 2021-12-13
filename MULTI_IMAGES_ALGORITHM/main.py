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
#import pymeanshift as pms

#import image
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
fil = filtering(im,9)
out = fil.routine()
cv2.imwrite('results/out4.png',out)

"""
Edge Detection and Hough Transform
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

hou = hough(out,2,186)
out = hou.routine()
cv2.imwrite('results/out5_2.png',out)


"""
Corner detection
"""
from corners import corner
im = cv2.imread("results/out5_2.png",0)
#im= cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
im = np.array(im,dtype="float32")
cor = corner(im,2,3,0.04,144)
out = cor.routine()

cv2.imwrite("results/out6.png",out)

"""
Finding protraits
"""
from finding import finding

img = cv2.imread('results/out5_1.png',0)
orig_img =cv2.imread('test.jpg')
fin_pain = finding(img,orig_img)
rects = fin_pain.routine_1()
det = fin_pain.routine_2(rects)
te = np.zeros(img.shape)
for i in rects:
    
    x,y,w,h = i
    te[ y:h+y,x:w+x] = 255 
cv2.imwrite('results/out7.png',te)
n = fin_pain.routine_3(rects)



"""
Final image matching and evaluation of our model
"""
from final_img_matching_and_evaluation import final_img_matching_and_evaluation,final_img_matching_and_evaluation_
imgs = ['results/ground/ground_truth_0.png','results/ground/ground_truth_1.png','results/ground/ground_truth_2.png','results/ground/ground_truth_3.png','results/ground/ground_truth_4.png','results/ground/ground_truth_5.png','results/ground/ground_truth_6.png','results/ground/ground_truth_7.png','results/ground/ground_truth_8.png']
detc = ['results/detected/detected_0.png','results/detected/detected_1.png','results/detected/detected_2.png','results/detected/detected_3.png','results/detected/detected_4.png','results/detected/detected_5.png','results/detected/detected_6.png','results/detected/detected_7.png','results/detected/detected_8.png']
fin1 = final_img_matching_and_evaluation(imgs,detc)
map_,acc = fin1.routine_1()
j = 0
for key in map_:
    a = cv2.imread(key)
    b = cv2.imread(map_[key])
    fin2 = final_img_matching_and_evaluation_(a,b)
    out = fin2.routine_1()
    fil = "results/match_"+str(j)+".jpg"
    cv2.imwrite(fil,out)
    j+=1
time2 = time.time()


print("################## Results ##################")
print("Time taken by our algorithm is: "+str(time2-time1)+" seconds")
print("Total number of protraits: ",n)
print("Our Average TM_CCOEFF_NORMED coefficient is %.2f"%(np.array(acc).mean()))
print("#################Conclusion##################")
print("for results and images please see results folder in the directory in which this file is present")
print("So our algorithm is promising and good enough")