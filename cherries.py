#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 09:47:24 2017

@author: bruno
"""

import cv2
print("Versión OpenCV: " + cv2.__version__)

import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('cherries.jpg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
mask = mask // 255

b,g,r = cv2.split(img)
b = b * mask
g = g * mask
r = r * mask
relevant = cv2.merge((b,g,r)).astype(np.uint8)

median = cv2.medianBlur(relevant, 15)

b,g,r = cv2.split(median)

mix = 0.9*r+0.1*g;
mix = mix.astype(np.uint8)

ret,otsu = cv2.threshold(mix,50,255,cv2.THRESH_BINARY)
# _,otsu = cv2.threshold(mix,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

kernel = np.ones((5,5),np.uint8)
closing = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel) // 255

prod = img * cv2.merge((1 - closing, 1 - closing, 1 - closing)).astype(np.uint8)
segmented = img * cv2.merge((closing, closing, closing)).astype(np.uint8)

"""
img_circles = img.copy()
# climage = cv2.cvtColor(segmented,cv2.COLOR_BGR2GRAY)
climage = (closing * 255)
circles = cv2.HoughCircles(climage,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=20,minRadius=0,maxRadius=0)
circles = np.uint16(np.around(circles))
if circles is not None:
  print(circles)
  for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(img_circles,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(img_circles,(i[0],i[1]),2,(0,0,255),3)
    
params = cv2.SimpleBlobDetector_Params()
 
# Change thresholds
params.minThreshold = 0;
params.maxThreshold = 255;
 
# Filter by Area.
params.filterByArea = False
params.minArea = 50
 
# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1
 
# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87
 
# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.7
 
detector = cv2.SimpleBlobDetector_create(params)
 
# Detect blobs.
reversemask = 255 - climage
keypoints   = detector.detect(reversemask)
print(len(keypoints))
im_with_keypoints = cv2.drawKeypoints(climage, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
"""

im2, contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

output = img.copy()
# output = cv2.merge((1 - closing, 1 - closing, 1 - closing)).astype(np.uint8) * 255

roi_list = []
hist_list = []

for index, cnt in enumerate(contours):
  x,y,w,h = cv2.boundingRect(cnt)
  cv2.rectangle(output,(x,y),(x+w,y+h),(255,255,0),2)
  
  roi = segmented[y:y+h, x:x+w]
  roi_list.append(roi)
  
  # hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
  hist = cv2.calcHist([roi], [2], None, [256], [0, 256])
  # Elimina los pixeles negros
  hist = np.delete(hist, [0])
  hist_list.append(hist)
  
  (x,y),radius = cv2.minEnclosingCircle(cnt)
  center = (int(x),int(y))
  radius = int(radius)
  cv2.circle(output,center,radius,(0,255,0),2)  
# Apply median filter to smooth image
#median = cv2.medianBlur(img, 31)
#ret,thresh = cv2.threshold(median,127,255,cv2.THRESH_BINARY)
#graythresh = cv2.cvtColor(thresh,cv2.COLOR_BGR2GRAY)


# Obtain binarized image
# ret,thresh1 = cv2.threshold(median,127,255,cv2.THRESH_BINARY)
# graythresh = cv2.cvtColor(thresh1,cv2.COLOR_BGR2GRAY)
# mask = graythresh * (1/255)
# Check if values are only 0 and 255
# unique, counts = np.unique(thresh1, return_counts=True)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Base image")
plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(cv2.cvtColor(relevant, cv2.COLOR_BGR2RGB))
plt.title("Relevant area in the image")
plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))
plt.title("Median filtering")
plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(b, cmap = 'gray')
plt.title("Blue Channel")
plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(g, cmap = 'gray')
plt.title("Green Channel")
plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(r, cmap = 'gray')
plt.title("Red Channel")
plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(mix, cmap = 'gray')
plt.title("Mixed 0.9R+0.1G Image")
plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(otsu, cmap = 'gray')
plt.title("Otsu Thresholding")
plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(closing, cmap = 'gray')
plt.title("Image after closing")
plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(cv2.cvtColor(prod, cv2.COLOR_BGR2RGB))
plt.title("Excluded pixels from image")
plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
plt.title("Segmented Image")
plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Resulting Detection")
plt.xticks([]), plt.yticks([])
plt.show()

for index, roi in enumerate(roi_list):
  plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
  plt.title("Detected Cherry N°: " + str(index))
  plt.xticks([]), plt.yticks([])
  plt.show()
  
  plt.plot(hist_list[index])
  plt.title("Histogram of Detected Cherry N°: " + str(index))
  plt.show()

"""
plt.imshow(cv2.cvtColor(img_circles, cv2.COLOR_BGR2RGB))
plt.title("Circle Detection")
plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title("Blob Detection")
plt.xticks([]), plt.yticks([])
plt.show()
"""
#plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
#plt.xticks([]), plt.yticks([])
#plt.show()

#plt.imshow(graythresh)
#plt.xticks([]), plt.yticks([])
#plt.show()
# cv2.imwrite("testing.jpg", median)

