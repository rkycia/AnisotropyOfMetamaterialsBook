#! /usr/bin/env python


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import cv2


#read file
image = cv2.imread("mandrill.png")

# convert image to gray
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("output/GrayMandrill.png", imageGray)


#make historgram of gray image
histGray= cv2.calcHist([imageGray],[0],None,[256],[0,256])
plt.plot(histGray/max(histGray))   
plt.xlabel("bin")
plt.ylabel("normalized cout")
plt.savefig("output/GrayHistogram.png")
plt.show()


#cumulative distribution
cumulativeSumGray = np.cumsum(histGray)
plt.plot(cumulativeSumGray/sum(histGray))   
plt.xlabel("bin")
plt.ylabel("normalized sum")
plt.savefig("output/GrayHistogramCumulative.png")
plt.show()


#equalization
imageGrayEqualized = cv2.equalizeHist(imageGray)
cv2.imwrite("output/GrayMandrillEqualized.png", imageGrayEqualized)

#make historgram of gray image
histGrayEqualized = cv2.calcHist([imageGrayEqualized],[0],None,[256],[0,256])
plt.plot(histGrayEqualized/max(histGrayEqualized))   
plt.xlabel("bin")
plt.ylabel("normalized cout")
plt.savefig("output/GrayHistogramEqualized.png")
plt.show()


#cumulative distribution
cumulativeSumGrayEqualized = np.cumsum(histGrayEqualized)
plt.plot(cumulativeSumGrayEqualized/sum(histGrayEqualized))   
plt.xlabel("bin")
plt.ylabel("normalized sum")
plt.savefig("output/GrayHistogramCumulativeEqualized.png")
plt.show()



#normalize color image
src = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(src)
v = cv2.equalizeHist(v)
src = cv2.merge([h,s,v])
imageEqualized = cv2.cvtColor(src, cv2.COLOR_HSV2BGR)
cv2.imwrite("output/MandarillEqualized.png", imageEqualized)
