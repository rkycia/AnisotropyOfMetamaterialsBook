#! /usr/bin/env python


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import cv2


#prepare figure
image = np.zeros((512,512,3), np.uint8)
image = image + 255
cv2.line(image,(0,0),(200,200),(0,0,0),2)
#filled rectangle
cv2.rectangle(image,(200,200),(400,400),(0,255,0),-1)
cv2.ellipse(image,(100,400),(100,50),0,0,270,(255,0,255),-1)
#generate random noise points
x = 512* np.random.rand(1000)
y = 512* np.random.rand(1000)
for i,j in zip(x,y):
    cv2.circle(image,(int(i),int(j)), 1, (0,0,0), -1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("output/initialImageMotphology.png", image)

#erosion
imageErode = cv2.erode(image, np.ones((3, 3), np.uint8))
cv2.imwrite("output/Erode.png", imageErode)

#dilation
imageDilate = cv2.dilate(image, np.ones((3, 3), np.uint8))
cv2.imwrite("output/Dilate.png", imageDilate)

#opening
imageOpening = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
cv2.imwrite("output/Opening.png", imageOpening)

#closing
imageClosing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
cv2.imwrite("output/Closing.png", imageClosing)

