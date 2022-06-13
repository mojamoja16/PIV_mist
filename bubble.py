import cv2 
#from google.colab.patches import cv2_imshow
import numpy as np
from matplotlib import pyplot as plt

path ='image_0000.jpg'
img = cv2.imread(path) # pathは画像を置いている場所を指定

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(img,(11,11),0)
bil=cv2.bilateralFilter(img,9,75,75)
hsv = cv2.cvtColor(bil, cv2.COLOR_BGR2HSV)


#img_gray = cv2.imread(hsv, 0)

threshold = 160
ret, img_thresh = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
'''
circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT,dp=1,minDist=70,
                            param1=100,param2=55,minRadius=10,maxRadius=0)
 
circles = np.uint16(np.around(circles))
 
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
'''

img_canny = cv2.Canny(hsv, 300, 100)
#img_sobel = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
#cv2_imshow(img_sobel)
#cv2.imshow(img_canny)
plt.imshow(hsv)
plt.show()