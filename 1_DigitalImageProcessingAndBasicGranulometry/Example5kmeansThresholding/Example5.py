#! /usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import cv2


img1 = cv2.imread("Picture2.png")



#cut out printout
img1 = img1[50:-80,10:]
cv2.imwrite("output/Example5Cut.png", img1)


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
plt.savefig("output/Example5ColorHistograms")
plt.show()



#histogram equalization
src = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(src)
#v = cv2.equalizeHist(v)
src = cv2.merge([h,s,v])
img1 = cv2.cvtColor(src, cv2.COLOR_HSV2BGR)
cv2.imwrite("output/Example5Equalization.png", img1)


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
plt.savefig("output/Example5ColorHistogramsEqualization.png")
plt.show()

xShape, yShape ,_ = img1.shape

data = img1.reshape(xShape * yShape, 3)

df = pd.DataFrame(data)
print(df)

#k-means number of clusters estimation
from sklearn.cluster import KMeans

k = range(1,10)
WSSE = []

for i in k:
    model = KMeans(n_clusters = i, n_init='auto')
    model.fit_predict(df)
    WSSE.append(model.inertia_)

plt.plot(k,WSSE)
plt.xlabel('K')
plt.ylabel('WSSE')
plt.grid(True)
plt.savefig("output/Example5kmeans.png")
plt.show()


#k-means
km = KMeans( n_clusters=3, n_init='auto')
labels = km.fit_predict(df)
#create mask from class labels
mask = labels.reshape(xShape, yShape, 1)
b,g,r = cv2.split(img1)


imgLabelled = []
masks = []
for label in range(3):
    maskClass = np.where(mask == label, 255, 0)
    maskClass = maskClass.astype(np.uint8)
    masks.append(maskClass)
    cv2.imwrite("output/mask{}.png".format(label), maskClass)
    imgLabel = cv2.bitwise_and(img1, img1, mask = maskClass)
    imgLabelled.append(imgLabel)
    cv2.imwrite("output/label{}.png".format(label), imgLabel)

                      
imgClass0 = imgLabelled[2]
#bilateral filtering
#imgClass0 = cv2.bilateralFilter(imgClass0,9,75,75)
cv2.imwrite("output/Example5BWFiltering.png", imgClass0)

gray = cv2.cvtColor(imgClass0, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)
#equalize
gray = cv2.equalizeHist(gray)
cv2.imwrite("output/Example5Class0BW.png", gray)
                      

#remove small dots
thresh1 = gray.copy()
thresh1 = cv2.dilate(thresh1, np.ones((3, 3), np.uint8))
thresh1 = cv2.erode(thresh1, np.ones((3, 3), np.uint8))        
cv2.imwrite("output/Example5Class0BWMorphology.png", thresh1)



#find contours
#contours, _ = cv2.findContours(thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1 )
contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1 )
blank = np.ones(thresh1.shape)*255
cv2.drawContours(blank, contours, -1, (0, 0, 0), 1)
print("# of contours found: ", len(contours))
cv2.imwrite("output/Example5BWCountour.png", blank)
  
            