#! -*- coding: utf-8 -*-

import cv2
import numpy as np

wname = "Face Detection"
cv2.namedWindow(wname)

face_classifier = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")

def face(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
  for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
  
  eyes = eye_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))
  for (x, y, w, h) in eyes:
    cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)

vid = cv2.VideoCapture("vid_test.mp4")
out = cv2.VideoWriter('output.avi', -1, 20.0, (1280, 720))
while vid.isOpened():
  succ, frame = vid.read()
  if not succ:
    break

  cv2.imshow(wname, frame)
  face(frame)
  out.write(frame)

  if cv2.waitKey(25) & 0xFF == ord("q"):
    break
vid.release()
out.release()

# cv2.waitKey(0)
cv2.destroyAllWindows()
