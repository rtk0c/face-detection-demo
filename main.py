#! -*- coding: utf-8 -*-

import cv2
import numpy as np

wname = "Face Detection"
cv2.namedWindow(wname)

img = cv2.imread("/Users/tianyiguan/Documents/Tinovation/FaceDetection/img_lights.jpg")
cv2.imshow(wname, img)

cv2.waitKey(0)
cv2.destroyAllWindows()
