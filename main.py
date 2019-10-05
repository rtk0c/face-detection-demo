#! -*- coding: utf-8 -*-

import cv2
import numpy as np

wname = "Face Detection"
cv2.namedWindow(wname)

# TODO actual image set
img = cv2.imread("/Users/tianyiguan/Documents/Tinovation/FaceDetection/img_lights.jpg")
# cv2.imshow(wname, img)

face_classifier = cv2.CascadeClassifier("Haarcascades/haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier ("Haarcascades/haarcascade_eye.xml")
def face_detector (img, size=0.5):
  gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
  faces = face_classifier.detectMultiScale(gray, 1.3, 5)
  if faces is ():
    return img

# TODO output

cv2.waitKey(0)
cv2.destroyAllWindows()
