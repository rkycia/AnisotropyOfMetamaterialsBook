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

#set reproducibility of initial image
np.random.seed(1)

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
cv2.imwrite("output/initialImageSegmentationColor.png", imageGray)


#prepare data for classification
xShape, yShape = imageGray.shape
data = imageGray.reshape(xShape * yShape, 1)
df = pd.DataFrame(data)


#k-means number of clusters estimation
from sklearn.cluster import KMeans


k = range(1,5)
WSSE = []

for i in k:
    model = KMeans(n_clusters = i, n_init='auto')
    model.fit_predict(df)
    WSSE.append(model.inertia_)

plt.plot(k,WSSE)
plt.xlabel('K')
plt.ylabel('WSSE')
plt.grid(True)
plt.savefig("output/Example5kmeansSegmentationColor.png")
plt.show()


#k-means
NClusters = 2

km = KMeans( n_clusters= NClusters, n_init='auto')
labels = km.fit_predict(df)
#create mask from class labels
mask = labels.reshape(xShape, yShape, 1)



imgLabelled = []
masks = []
for label in range(NClusters):
    maskClass = np.where(mask == label, 255, 0)
    maskClass = maskClass.astype(np.uint8)
    masks.append(maskClass)
    cv2.imwrite("output/SegmentationColorMask{}.png".format(label), maskClass)
    imgLabel = cv2.bitwise_and(imageGray, imageGray, mask = maskClass)
    imgLabelled.append(imgLabel)
    cv2.imwrite("output/SegmentationColorLabel{}.png".format(label), imgLabel)

  
            