#! /usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import cv2


img1 = cv2.imread("Picture1.png")

#cut out printout
img1 = img1[50:-10,10:]
cv2.imwrite("output/Example1Cut.png", img1)


#histograms of colors
hist= cv2.calcHist([img1],[0],None,[256],[0,256])
hist2=cv2.calcHist([img1],[1],None,[256],[0,256])
hist3=cv2.calcHist([img1],[2],None,[256],[0,256])
##plot histograms
plt.subplot(1, 3, 1)
plt.plot(hist/max(hist))   
plt.title('Hist Blue : ')
plt.subplot(1, 3, 2)
plt.plot(hist2/max(hist2),color="green")   
plt.title('Hist Green : ')
plt.subplot(1, 3, 3)
plt.plot(hist3/max(hist3),color="red")   
plt.title('Hist Red : ')
plt.savefig("output/Example1ColorHistograms")
plt.show()



#histogram equalization
src = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(src)
v = cv2.equalizeHist(v)
src = cv2.merge([h,s,v])
img1 = cv2.cvtColor(src, cv2.COLOR_HSV2BGR)
cv2.imwrite("output/Example1Equalization.png", img1)


#histograms of colors after normalization
hist= cv2.calcHist([img1],[0],None,[256],[0,256])
hist2=cv2.calcHist([img1],[1],None,[256],[0,256])
hist3=cv2.calcHist([img1],[2],None,[256],[0,256])
plt.subplot(1, 3, 1)
plt.plot(hist/max(hist))   
plt.title('Hist Blue : ')
plt.subplot(1, 3, 2)
plt.plot(hist2/max(hist2),color="green")   
plt.title('Hist Green : ')
plt.subplot(1, 3, 3)
plt.plot(hist3/max(hist3),color="red")   
plt.title('Hist Red : ')
plt.savefig("output/Example1ColorHistogramsEqualization.png")
plt.show()




#scale figure
img1 = cv2.resize(img1,None,fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
# convert to BW
img1_BW = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
cv2.imwrite("output/Example1BW.png", img1_BW)


#bilateral filtering
img1_BW = cv2.bilateralFilter(img1_BW,9,75,75)
cv2.imwrite("output/Example1BWFiltering.png", img1_BW)


#compute gray histogram
hist1 = cv2.calcHist([img1_BW], [0], None, [256], [0,256])
plt.plot(hist1.ravel()/hist1.max())
plt.savefig("output/Example1HistogramBW.png")
plt.show()

#tresholding - we select 40 for selecting black granules
ret1m, thresh1 = cv2.threshold(img1_BW, 40, 256, cv2.THRESH_BINARY)

#remove small dots
thresh1 = cv2.dilate(thresh1, np.ones((5, 5), np.uint8))
thresh1 = cv2.erode(thresh1, np.ones((5, 5), np.uint8))

cv2.imwrite("output/Example1BWThresholding.png", thresh1)

#reverse colors in mask
thresh2 = abs(thresh1 - 255)


#find contours
contours, _ = cv2.findContours(thresh2, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1 )
blank = np.ones(thresh2.shape)*255
cv2.drawContours(blank, contours, -1, (0, 0, 0), 3)
print("# of contours found: ", len(contours))
cv2.imwrite("output/Example1BWCountour.png", blank)




#minmimal area rectangle
img1_tmp1 = img1.copy()
img1_tmp2 = img1.copy()
for idx, cnt in enumerate(contours):
    minRect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(minRect)
    box = np.int0(box)
    cv2.drawContours(img1_tmp1, [box], 0, (0,0,255), 2)
    cv2.drawContours(img1_tmp2, [box], 0, (0,0,255), 2)
    cv2.putText(img1_tmp2, '{}'.format(idx), (int(box[0][0]), int(box[0][1]) ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) 
cv2.imwrite("output/Example1BWMinAreaRectangle.png", img1_tmp1)
cv2.imwrite("output/Example1BWMinAreaRectangleWithIndices.png", img1_tmp2)



#fit ellipses
blank = np.ones(thresh2.shape)*255
ellipsesData = []
img_tmp = img1.copy()
for cnt in contours:
    #if less points preventing fitting ellipse
    if cnt.size < 10 :
        continue
    x,y,w,h = cv2.boundingRect(cnt)
    #remove small conoturs
    if w < 2 or h < 2:
        continue
    ellipse = cv2.fitEllipse(cnt)
    cv2.ellipse(blank, ellipse, (0,255, 255), 1, cv2.LINE_AA)
    cv2.ellipse(img_tmp, ellipse, (0,255, 255), 1, cv2.LINE_AA)
    ellipsesData.append({"id": idx, "x":ellipse[0][0], "y":ellipse[0][1], "majorAxis": ellipse[1][0], "minorAxis": ellipse[1][1], "angle": ellipse[2]})
    
cv2.imwrite("output/Example1BWEllipses.png", blank)
cv2.imwrite("output/Example1BWEllipsesColor.png", img_tmp)

df_ellipses = pd.DataFrame.from_records(ellipsesData)
df_ellipses.to_csv("output/ellipses.csv")

plt.hist(np.array(df_ellipses.angle)%90)
plt.xlabel("angle[deg]  mod 90")
plt.ylabel("No. of ellipses")
plt.savefig("output/Example1AngleDistibution.png")
plt.show()






