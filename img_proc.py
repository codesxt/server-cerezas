#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 14:37:42 2017

@author: bruno
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

DEBUG = False

def count_cherries(filename, out_file):
  img = cv2.imread(filename)

  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

  # noise removal
  kernel = np.ones((3,3),np.uint8)
  opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

  # sure background area
  sure_bg = cv2.dilate(opening,kernel,iterations=3)

  # Finding sure foreground area
  dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
  ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

  # Finding unknown region
  sure_fg = np.uint8(sure_fg)
  unknown = cv2.subtract(sure_bg,sure_fg)

  # Marker labelling
  ret, markers = cv2.connectedComponents(sure_fg)
  # Add one to all labels so that sure background is not 0, but 1
  markers = markers+1
  # Now, mark the region of unknown with zero
  markers[unknown==255] = 0

  markers = cv2.watershed(img,markers)
  img[markers == -1] = [255,0,0]

  params = cv2.SimpleBlobDetector_Params()

  # Change thresholds
  params.minThreshold = 10;
  params.maxThreshold = 255;

  # Filter by Area.
  params.filterByArea = True
  params.minArea = 10

  # Filter by Circularity
  params.filterByCircularity = True
  params.minCircularity = 0.1

  # Filter by Convexity
  params.filterByConvexity = False
  params.minConvexity = 0.87

  # Filter by Inertia
  params.filterByInertia = True
  params.minInertiaRatio = 0.01

  detector = cv2.SimpleBlobDetector_create(params)

  keypoints = detector.detect(abs(255-sure_fg))
  cv2.imwrite(out_file, sure_fg)
  return(len(keypoints))

def rectify_image(marker_file, marker_size, scene_file):
  # marker_file : Nombre del archivo de marcador
  # marker_size : Tamaño del lado del marcador en centímetros
  # scene_file  : Archivo de la foto sin rectificar

  # Lee el marcador en blanco y negro
  img_marker = cv2.imread(marker_file,0)

  # Se inicializa el detector/descriptor ORB
  orb = cv2.ORB_create()
  kp_marker = orb.detect(img_marker,None)
  kp_marker, des_marker = orb.compute(img_marker, kp_marker)

  # Grafica la imagen del marcador con los keypoints detectados
  if DEBUG:
    img_marker_with_kp = cv2.drawKeypoints(img_marker, kp_marker, None, color=(0,255,0), flags=0)
    plt.imshow(img_marker_with_kp), plt.show()

  scene_original = cv2.imread(scene_file) # Imagen en Color
  scene = cv2.imread(scene_file,0)        # Imagen en blanco y negro para la detección

  # Se hace la detección y se calculan los descriptores
  kp_scene = orb.detect(scene, None)
  kp_scene, des_scene = orb.compute(scene, kp_scene)

  # Se muestran los keypoints detectados en la escena
  if DEBUG:
    img_scene_with_kp = cv2.drawKeypoints(scene, kp_scene, None, color=(0,255,0), flags=0)
    plt.imshow(img_scene_with_kp), plt.show()

  # Configura el matcher de descriptores FLANN
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
  if len(good)>=MIN_MATCH_COUNT:
      src_pts = np.float32([ kp_marker[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
      dst_pts = np.float32([ kp_scene[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

      M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
      M2, mask2 = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)
      matchesMask = mask.ravel().tolist()

      if M is None:
          print("Homography failed. This error sometimes happens for large images.")
          matchesMask = None
          return None, 0

      h,w = img_marker.shape
      pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

      dst = cv2.perspectiveTransform(pts,M)
      # Width of detected square marker in the scene
      sq_width = np.linalg.norm(dst[0][0]-dst[1][0])

      scene = cv2.polylines(scene,[np.int32(dst)],True,255,3, cv2.LINE_AA)

  else:
      print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
      matchesMask = None
      return None, 0

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
  img_rect = cv2.warpPerspective(scene_original, M2, (int(width*scale), int(height*scale)))
  #scene = cv2.resize(scene,(int(scale*width), int(scale*height)), interpolation = cv2.INTER_LINEAR)
  if DEBUG:
    plt.imshow(img_rect),plt.show()

  MARKER_SIZE = marker_size # CMS
  pix_scale = MARKER_SIZE / img_marker.shape[0]
  return img_rect, pix_scale

def segment_cherries(image_file, output_file):
  img = image_file

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

  kernel = np.ones((10,10),np.uint8)
  closing = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel) // 255

  segmented = img * cv2.merge((closing, closing, closing)).astype(np.uint8)

  im2, contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

  output = img.copy()
  # output = cv2.merge((1 - closing, 1 - closing, 1 - closing)).astype(np.uint8) * 255

  roi_list    = []
  hist_list   = []
  radius_list = []

  for index, cnt in enumerate(contours):
    if cv2.contourArea(cnt) < 1000:
      continue

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
    radius_list.append(radius)

  cv2.imwrite(output_file, output)
  return roi_list, hist_list, radius_list
