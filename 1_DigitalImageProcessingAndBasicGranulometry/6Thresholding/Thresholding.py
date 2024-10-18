#! /usr/bin/env python


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import cv2


#prepare figure
image = np.zeros((512,512,3), np.uint8)
image = image + 255

#perturbation
cv2.rectangle(image,(100,0),(500,400),(240,240,240),-1)
cv2.rectangle(image,(0,100),(400,500),(0,240,240),-1)

#generate class 1
x = 512* np.random.rand(200)
y = 512* np.random.rand(200)
for i,j in zip(x,y):
    cv2.circle(image,(int(i),int(j)), 5, (100,100,100), -1)
    
#generate class 2    
x = 512* np.random.rand(50)
y = 512* np.random.rand(50)
for i,j in zip(x,y):
    cv2.circle(image,(int(i),int(j)), 20, (10,10,10), -1)

    

imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("output/initialImageThresholding.png", imageGray)


#make historgram of gray image
histGray= cv2.calcHist([imageGray],[0],None,[256],[0,256])
plt.plot(histGray/max(histGray))   
plt.xlabel("bin")
plt.ylabel("normalized cout")
plt.savefig("output/GrayHistogramThresholding.png")
plt.show()


#thresholding - manual
##first set
_ , mask1 = cv2.threshold(imageGray, 50, 255, cv2.THRESH_BINARY)
cv2.imwrite("output/MaskThresholding1.png", mask1)

##second set
_ , mask2 = cv2.threshold(cv2.bitwise_xor(imageGray, cv2.bitwise_not(mask1)) , 150, 255, cv2.THRESH_BINARY)
#mask2 = cv2.bitwise_not(cv2.inRange(imageGray, 50, 150))
cv2.imwrite("output/MaskThresholding2.png", mask2)


#thresholding - Otsu
_, imageOtsu = cv2.threshold(imageGray, 0, 255, cv2.THRESH_OTSU)
cv2.imwrite("output/MaskThresholdingOtsu.png", imageOtsu)


#thresholding - adaptive mean
imageAdaptiveMean = cv2.adaptiveThreshold(imageGray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,51,1)
cv2.imwrite("output/MaskThresholdingAdaptiveMean.png", imageAdaptiveMean)

#thresholding - adaptive gaussian
imageAdaptiveGaussian = cv2.adaptiveThreshold(imageGray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,51,1)
cv2.imwrite("output/MaskThresholdingAdaptiveGaussian.png", imageAdaptiveGaussian)
