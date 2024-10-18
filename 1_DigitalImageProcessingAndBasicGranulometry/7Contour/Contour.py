#! /usr/bin/env python


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import cv2


#prepare figure
image = np.zeros((512,512,3), np.uint8)
image = image + 255

    
#generate granules  
np.random.seed(1)  
x = 512* np.random.rand(50)
y = 512* np.random.rand(50)
for i,j in zip(x,y):
    cv2.circle(image,(int(i),int(j)), 20, (255,0,0), -1)

cv2.imwrite("output/initialImageContours.png", image)  

#convert to gray
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imageGray = cv2.bitwise_not(imageGray)
cv2.imwrite("output/maskImageContours.png", imageGray)

#erode if needed to separate contours
#imageErode = cv2.erode(imageGray, np.ones((3, 3), np.uint8))
#cv2.imwrite("output/ErodeImageContours.png", imageErode)

#find contours
contours, _ = cv2.findContours(imageGray, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1 )

#draw contours
imageGrayContours =  image.copy()
#imageGrayContours =  np.zeros(image.shape)+255
cv2.drawContours(imageGrayContours, contours, -1, (0, 0, 255), 3)
cv2.imwrite('output/contoursImageContours.png', imageGrayContours)


#Hough transform circles, see, e.g.,  https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html

img = imageGray.copy()
colorimg = image.copy()

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT, 1, 20, param1=20, param2=10, minRadius=2, maxRadius=50)
 
circles = np.uint16(np.around(circles))
for circle in circles[0,:]:
    # draw the outer circle
    cv2.circle(colorimg,(circle[0],circle[1]),circle[2],(0,0,255),2)
    # draw the center of the circle
    cv2.circle(colorimg,(circle[0],circle[1]),3,(0,0,0),3)

cv2.imwrite('output/HoughCirclesDetectorImageContours.png', colorimg)


#bouding rectangle
img = image.copy()
for idx, contour in enumerate(contours):
           
    #draw minimum area rectangle
    minRect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(minRect)
    box = np.int0(box)
    cv2.drawContours(img, [box], -1, (0,0,255), 1)
    #enumarate contours
    cv2.putText(img, '{}'.format(idx), (box[0][0], box[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA) 
    
    #extract image from bounding rectangle and write them to files
    x, y, w, h = cv2.boundingRect(contour)
    cropped = image[y:y + h, x:x + w, :].copy()
    feature = cv2.resize(cropped, (20, 20))
    cv2.imwrite("./output/figs/{}.png".format(idx), feature)
    
cv2.imwrite('output/MinAreaRectangleImageContours.png', img)


#fit ellipses
img = image.copy()
ellipses = [] #list for storing ellipsis parameters
for idx, contour in enumerate(contours):
    if contours[idx].size < 10 :
        continue
    ellipse = cv2.fitEllipse(contour) # ellypse is ((x,y), (majorAxis, minorAxis), angle)
    #remove bigger ellipses
    if max(ellipse[1]) < 200:
        ellipses.append({"id": idx, "x":ellipse[0][0], "y":ellipse[0][1], "majorAxis": ellipse[1][0], "minorAxis": ellipse[1][1], "angle": ellipse[2]})
        cv2.ellipse(img, ellipse, (0,0,255), 3, cv2.LINE_AA)
        cv2.putText(img, '{}'.format(idx), (int(ellipse[0][0]), int(ellipse[0][1]) ), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA) 
#save parameters of ellipses        
df_ellipses = pd.DataFrame.from_records(ellipses)
df_ellipses.to_csv("output/ellipses.csv")
#write image with elipses
cv2.imwrite('output/EllipsesImageContours.png', img)



#calculate moments for counturs
import math
huMomentsList = []
for idx, contour in enumerate(contours):
    #calculate moments
    moments = cv2.moments(contour)
    print("contour nr ", idx, ", moments:", moments)
    #calculate Hu moments 
    huMoments = cv2.HuMoments(moments)
    #scaling Hu moments
    for i in range(0,7):
       if huMoments[i] == 0:
           huMoments[i] = 0.0
       else:
           huMoments[i] = -1* math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))
        
    huMomentsList.append({'h0':huMoments[0], 'h1':huMoments[1], 'h2':huMoments[2], 'h3':huMoments[3], 'h4':huMoments[4], 'h5':huMoments[5], 'h6':huMoments[6]})

df_HuMoments = pd.DataFrame.from_records(huMomentsList)

df_HuMoments.to_csv("output/HuMoments.csv")







