#! /usr/bin/env python


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import cv2


#create gray image
image = np.zeros((512,512,3), np.uint8)

#reverse colors
image = image + 255

#line
cv2.line(image,(0,0),(200,200),(255,0,255),2)

#rectangle with line widt 2
cv2.rectangle(image,(200,200),(300,300),(0,255,255),2)

#filled rectangle (linewidth = -1)
cv2.rectangle(image,(300,300),(400,400),(0,0,0),-1)

#circle
cv2.circle(image,(400,100), 50, (0,0,255), 2)

#ellipse
cv2.ellipse(image,(100,400),(100,50),0,0,270,(255,0,255),-1)

#text
cv2.putText(image,'Example Text', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)


cv2.imwrite("output/drawing.png", image)