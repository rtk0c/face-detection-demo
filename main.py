#! -*- coding: utf-8 -*-

import cv2
import numpy as np
from config import *

print("Eye type: ", eye)
print("Face image path: ", img_path)

wname = "Face Detection"
cv2.namedWindow(wname)

img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_classifier = cv2.CascadeClassifier(face_types[face])
eye_classifier = cv2.CascadeClassifier(eye_types[eye])

faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
for (x, y, w, h) in faces:
  cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

eyes = eye_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))
for (x, y, w, h) in eyes:
  cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)

cv2.imshow(wname, img)


cv2.waitKey(0)
cv2.destroyAllWindows()
