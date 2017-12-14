#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:05:39 2017

@author: bruno
"""

# Updated OpenCV 3.1.0 to 3.3.0 with: conda install -c conda-forge opencv 

import numpy as np
import cv2
from matplotlib import pyplot as plt

DEBUG = False

img_marker = cv2.imread('marker.png',0)
orb = cv2.ORB_create()
kp_marker = orb.detect(img_marker,None)
kp_marker, des_marker = orb.compute(img_marker, kp_marker)
if DEBUG:
  img_marker_with_kp = cv2.drawKeypoints(img_marker, kp_marker, None, color=(0,255,0), flags=0)
  plt.imshow(img_marker_with_kp), plt.show()

scene = cv2.imread('marker_scene.png',0)
kp_scene = orb.detect(scene, None)
kp_scene, des_scene = orb.compute(scene, kp_scene)
if DEBUG:
  img_scene_with_kp = cv2.drawKeypoints(scene, kp_scene, None, color=(0,255,0), flags=0)
  plt.imshow(img_scene_with_kp), plt.show()

# Configures FLANN descriptor matcher
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

des_marker = np.float32(des_marker)
des_scene = np.float32(des_scene)

matches = flann.knnMatch(des_marker,des_scene,k=2)

good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:        
        good.append(m)
        
MIN_MATCH_COUNT=10
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp_marker[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp_scene[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    M2, mask2 = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img_marker.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    # Width of detected square marker in the scene
    sq_width = np.linalg.norm(dst[0][0]-dst[1][0])

    scene = cv2.polylines(scene,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None
    
if DEBUG:
  draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                 singlePointColor = None,
                 matchesMask = matchesMask, # draw only inliers
                 flags = 2)
  img_with_matches = cv2.drawMatches(img_marker,kp_marker,scene,kp_scene,good,None,**draw_params)
  plt.imshow(img_with_matches, 'gray'),plt.show()


h,w = img_marker.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)
scene = cv2.cvtColor(scene, cv2.COLOR_GRAY2RGB)
cv2.polylines(scene,[np.int32(dst)],True,(0,255,255), 5, cv2.LINE_4)

height, width = scene.shape[:2]
scale = img_marker.shape[0] / sq_width
img_rect = cv2.warpPerspective(scene, M2, (scene.shape[1], scene.shape[0]))
scene = cv2.resize(scene,(int(scale*width), int(scale*height)), interpolation = cv2.INTER_LINEAR)
plt.imshow(img_rect),plt.show()

MARKER_SIZE = 3 # CMS
pix_scale = MARKER_SIZE / img_marker.shape[0]

# Retornar: Imagen Rectificada y escala de Pixel
