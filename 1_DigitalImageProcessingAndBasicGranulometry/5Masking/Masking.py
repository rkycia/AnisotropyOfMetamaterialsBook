#! /usr/bin/env python


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import cv2


#prepare mask
maskBGR = np.zeros((512,512,3), dtype='uint8')
cv2.circle(maskBGR,(200,100),100,(255,255,255),-1)
mask = cv2.cvtColor(maskBGR, cv2.COLOR_BGR2GRAY)
cv2.imwrite("output/Mask.png", mask)


#read image
image = cv2.imread("mandrill.png")


#bitwise AND
maskedImageAND = cv2.bitwise_and(image, image, mask=mask)
cv2.imwrite("output/imageMaskedAnd.png", maskedImageAND)
##reverse backround
maskedImageAND[mask == 0] = 255
cv2.imwrite("output/imageMaskedAndReversedBackground.png", maskedImageAND)


#add two images
dst = cv2.add(maskBGR,image)
cv2.imwrite("output/imageAdd.png", dst)
